"""Drama 3 のエンジン（drama3/kernels.py）を公式の実装と比べる。イメージ sokutou/drama3-dev の中で動かす。

  PYTHON=/opt/fish-speech/.venv/bin/python IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh \
    drama3/check.py <確かめ方> [--ref dev/ref0.wav --ref-text ...]

確かめ方:
  layers   層ごと: 公式の層を float32 にした層（FP8 のエキスパートは float32 に戻す）の 1 トークンの出力を基準にして、float32
           のカーネル、bf16 のカーネル、公式の bf16 の層を SNR で比べる。backbone の 40 層はプロンプトの最後の行で、tower の
           13 層は最後の行と最後の音声の行（--ref のときだけある。生成のときはこちらの経路を通る）で確かめる（float32 の
           カーネルが実装の誤りを、bf16 が精度の落ち方を見る）
  teacher  教師強制: 公式の出力のフレーム（--teacher、drama3/bench.py --save-dir の .npy）を入れて、1 フレームのロジット
           （main token と 9 個のコード）を、公式（eager）とエンジン（グラフなし）で比べる。公式の残差を float32 にした設定
           （公式の fp16 のときの経路）と公式の差を、公式の精度の揺れの目安にする
  sampler  サンプリング: 決まったロジットで 4 万回ずつ引き、公式の規則（top-k・top-p・最上位は残す、RAS の引き直し、eos は
           引き直さない）から計算した確率とカイ二乗で比べる
  prefill  最初の入力: CUDA Graph（同じ長さの別のプロンプトで記録してから流す）と、参照の使い回し（prefix_cache）の
           隠れ状態・GDN の状態・注意の KV・tower の KV を、公式どおりの eager と比べる。公式どうしの差（区切りを 64 行に
           したとき、残差を float32 にしたとき）も出し、使い回しの差の目安にする。--ref がいる

参照（--ref、runs/ からの相対のパス）は、drama3/bench.py と drama3/serve.py と同じく WAV をコーデックで符号化して使う。
teacher で参照を使うときは、--teacher も同じ参照で作った出力（例: bench/drama3/wav-official-ref-1/official-ref-1）にする。
出力: runs/check/drama3/<確かめ方>[-<wd>][-ref].json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernels as K  # noqa: E402
from bench import REPO, SAMPLE_RATE, checkpoint_path  # noqa: E402

DEV = torch.device("cuda:0")
SENTENCES = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
LAYER_TEXT = "雨の日は家で本を読むのが好きです。"


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    e = float(((ref - got) ** 2).sum())
    return math.inf if e == 0 else 10 * math.log10(float((ref ** 2).sum()) / e)


def encode_reference(codec, rel: str) -> tuple[torch.Tensor, float]:
    import soundfile as sf

    p = Path(rel) if Path(rel).is_absolute() else REPO / "runs" / rel
    audio, sr = sf.read(p, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(1))
    if sr != SAMPLE_RATE:
        import torchaudio

        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    with torch.inference_mode():
        x = audio.to(DEV, dtype=next(codec.parameters()).dtype)[None, None]
        codes = codec.encode(x, torch.tensor([x.shape[-1]], device=DEV))[0][0]
    return codes.cpu(), len(audio) / SAMPLE_RATE


# ---------------------------------------------------------------- layers


def check_layers(args, model, ref) -> dict:
    from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer
    from fish_speech.models.text2semantic.fish_omni.generate import (DRAMA3_GRAPH_CAPACITY, build_drama3_prompt,
                                                                    ensure_drama3_caches)
    from fish_speech.models.text2semantic.fish_omni.layers import LayerContext
    from fish_speech.models.text2semantic.fish_omni.moe import dequantize_fp8_blocks
    from fish_speech.models.text2semantic.fish_omni.tower import TowerLayer

    f32 = torch.float32
    bb, tw, cfg = model.backbone, model.audio_decoder, model.config
    big = DRAMA3_GRAPH_CAPACITY
    ensure_drama3_caches(model, big)
    rt, rc = ref
    prompt = build_drama3_prompt(LAYER_TEXT, model.tokenizer, cfg, rt, rc)
    R = len(prompt)
    frames = prompt.to(DEV)
    # 公式の最初の入力を流し、各層の入力を拾う
    bin_, tin = {}, {}
    hooks = [layer.register_forward_pre_hook(lambda m, a, i=i: bin_.setdefault(i, []).append(a[0].detach().clone()))
             for i, layer in enumerate(bb.model.layers)]
    tl_all = [*tw.shared_layers, *tw.residual_layers]
    hooks += [layer.register_forward_pre_hook(
        lambda m, a, i=i: tin.setdefault(i, []).append([t.detach().clone() if torch.is_tensor(t) else t for t in a]))
        for i, layer in enumerate(tl_all)]
    with torch.inference_mode():
        positions = torch.arange(R, device=DEV)
        for s0 in range(0, R, 128):
            e0 = min(s0 + 128, R)
            part = frames[s0:e0]
            hidden = bb(inputs_embeds=tw.embed_backbone(part, bb.embed(part[:, 0]))[None], input_pos=positions[s0:e0])[0]
            pe = min(e0, R - 1)
            if pe > s0:
                tw.prefill(hidden[:pe - s0], positions[s0:pe], frames[s0:pe], frames[s0 + 1:pe + 1])
    for h in hooks:
        h.remove()
    print(f"rows {R}", flush=True)

    CAP = 256
    theta = float(cfg.text.rope_theta)
    rope32_bb, rope32_tw = K.rope_tables(CAP, DEV, theta)
    rope_bb, rope_tw = K.rope_tables(big, DEV, theta)
    k32, k16 = K.Kern("f32"), K.Kern("bf16")
    B32, B16 = K.Bufs(DEV), K.Bufs(DEV)
    st = k16.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    freq = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=f32, device=DEV) / 64))

    def ctx(pos, kv_len, cont):
        ang = pos.float()[:, None] * freq[None]
        ang = torch.cat((ang, ang), -1)
        return LayerContext(input_pos=pos, cos=ang.cos(), sin=ang.sin(), kv_len=kv_len, continue_state=cont, use_cache=True)

    def states(m, full):
        return [m.k_cache.clone(), m.v_cache.clone()] if full else [m.conv_state.clone(), m.recurrent_state.clone()]

    def put(m, full, sts, n=None):
        if full:
            m.k_cache.zero_()
            m.v_cache.zero_()
            m.k_cache[:, :, :n].copy_(sts[0][:, :, :n])
            m.v_cache[:, :, :n].copy_(sts[1][:, :, :n])
        else:
            m.conv_state.copy_(sts[0])
            m.recurrent_state.copy_(sts[1])

    rows = []
    with torch.inference_mode():
        for li in range(len(bb.model.layers)):
            src = bb.model.layers[li]
            full = src.layer_type == "full_attention"
            L32 = DecoderLayer(cfg.text, li, fp8=False, dtype=f32, device="meta")
            L32.to_empty(device=DEV)
            p32 = dict(L32.named_parameters())
            for n_, t in src.state_dict().items():
                if ".experts." not in n_:
                    p32[n_].data.copy_(t.float())
            ex = src.mlp.experts
            w13, w2 = L32.mlp.experts.w13_weight.data, L32.mlp.experts.w2_weight.data
            for e in range(0, w13.shape[0], 32):
                w13[e:e + 32] = dequantize_fp8_blocks(ex.w13_weight[e:e + 32], ex.w13_scale_inv[e:e + 32], f32)
                w2[e:e + 32] = dequantize_fp8_blocks(ex.w2_weight[e:e + 32], ex.w2_scale_inv[e:e + 32], f32)
            mx32 = L32.self_attn if full else L32.linear_attn
            if full:
                mx32.allocate_cache(1, CAP, f32, DEV)
            else:
                mx32.allocate_cache(1, f32, DEV)
            x = torch.cat(bin_[li], 1)[0]
            x32 = x.float()
            L32(x32[None, :R - 1], ctx(torch.arange(R - 1, device=DEV), R - 1, False))
            pre = states(mx32, full)
            yref = L32(x32[None, R - 1:R], ctx(torch.tensor([R - 1], device=DEV), R, True))[0, 0]
            post = states(mx32, full)
            put(mx32, full, pre, R - 1)
            pos = torch.tensor([R - 1], dtype=torch.int64, device=DEV)
            S32 = K.backbone_layer_spec(L32, "f32", B32, DEV, experts=ex)
            B32.x.copy_(x32[R - 1])
            with st:
                K.run_backbone_layer(k32, S32, B32, pos, 0, rope32_bb, CAP)
            torch.cuda.synchronize()
            y32 = B32.x.clone()
            s32 = states(mx32, full)
            # bf16 のカーネル（モデルのキャッシュに、float32 の層で最後の行の手前まで流した状態を入れる）
            mx16 = src.self_attn if full else src.linear_attn
            put(mx16, full, pre, R - 1)
            S16 = K.backbone_layer_spec(src, "bf16", B16, DEV)
            B16.x.copy_(x32[R - 1])
            with st:
                K.run_backbone_layer(k16, S16, B16, pos, 0, rope_bb, big)
            torch.cuda.synchronize()
            y16 = B16.x.clone()
            put(mx16, full, pre, R - 1)
            yoff = src(x[None, R - 1:R], ctx(torch.tensor([R - 1], device=DEV), R, True))[0, 0].float()
            d = yref - x32[R - 1]
            r = {"layer": li, "type": "full" if full else "gdn", "f32": snr_db(d, y32 - x32[R - 1]),
                 "bf16": snr_db(d, y16 - x32[R - 1]), "official_bf16": snr_db(d, yoff - x32[R - 1])}
            if full:
                r["kv_f32"] = min(snr_db(post[0][0, :, R - 1], s32[0][0, :, R - 1]),
                                  snr_db(post[1][0, :, R - 1], s32[1][0, :, R - 1]))
            else:
                r["state_f32"] = min(snr_db(post[0], s32[0]), snr_db(post[1], s32[1]))
            rows.append(r)
            print(r, flush=True)
            del L32, w13, w2, mx32, S32, S16, p32
            gc.collect()
            torch.cuda.empty_cache()
        # tower
        cb32 = tw.codebook_embeddings.weight.float()
        for j in range(len(tl_all)):
            src = tl_all[j]
            shared = j < len(tw.shared_layers)
            a = [torch.cat([c[i] for c in tin[j]], 0) if torch.is_tensor(tin[j][0][i]) else None
                 for i in range(len(tin[j][0]))]
            hid, codes, sem, tpos = a[0], a[1], a[2], a[3]
            fr = frames[tpos]
            unseen = None
            if j == 0:  # 最初の共有の層の shared_input を、入力のフレームから float32 で作り直す
                idx = fr[:, 3:11] + torch.arange(2, 10, device=DEV) * 4096
                unseen = cb32[idx].sum(1) * tw.is_semantic(fr[:, 0])[:, None]
            # 最後の行（テキスト）と、最後の音声の行（semantic。生成するフレームはすべてこちら）で確かめる
            qs = [len(tpos) - 1] + ([int(torch.nonzero(sem).max())] if bool(sem.any()) else [])
            for q in dict.fromkeys(qs):
                T32 = TowerLayer(cfg.audio_decoder, shared=shared, dtype=f32, device=DEV)
                T32.load_state_dict({k_: v.float() for k_, v in src.state_dict().items()})
                at32 = T32.transformer_layer.attention
                at32.setup_cache(CAP)
                T32(hid[:q].float(), codes[:q], sem[:q], tpos[:q], None if unseen is None else unseen[:q])
                pre = [at32.k_cache.clone(), at32.v_cache.clone()]
                yref = T32(hid[q:q + 1].float(), codes[q:q + 1], sem[q:q + 1], tpos[q:q + 1],
                           None if unseen is None else unseen[q:q + 1])[0]
                post = [at32.k_cache.clone(), at32.v_cache.clone()]
                at32.k_cache.copy_(pre[0])
                at32.v_cache.copy_(pre[1])
                p = int(tpos[q])
                posT = torch.tensor([p], dtype=torch.int64, device=DEV)
                fo = torch.zeros(K.WIDTH, dtype=torch.int64, device=DEV)
                fo[0] = K.SEM_BEGIN if bool(sem[q]) else K.EOS
                fo[1] = codes[q]
                fi_ = fr[q].clone()
                T = K.tower_layer_spec(T32, "f32", B32, DEV)
                B32.tr.zero_()
                hsrc = hid[q].float().contiguous()
                with st:
                    K.run_tower_layer(k32, T, B32, hsrc, None, fo, 1, cb32 if j == 0 else None, fi_, posT, rope32_tw, CAP)
                torch.cuda.synchronize()
                tr = B32.tr
                y32 = tr * torch.rsqrt(tr.pow(2).mean() + 1e-6) * T["fnw"]
                kv32 = min(snr_db(post[0][0, :, p], at32.k_cache[0, :, p]), snr_db(post[1][0, :, p], at32.v_cache[0, :, p]))
                at16 = src.transformer_layer.attention
                at16.k_cache.zero_()
                at16.v_cache.zero_()
                at16.k_cache[:, :, :CAP].copy_(pre[0])
                at16.v_cache[:, :, :CAP].copy_(pre[1])
                T16 = K.tower_layer_spec(src, "bf16", B16, DEV)
                with st:
                    K.run_tower_layer(k16, T16, B16, hsrc, None, fo, 1, tw.codebook_embeddings.weight if j == 0 else None, fi_,
                                      posT, rope_tw, big)
                torch.cuda.synchronize()
                tr = B16.tr
                y16 = tr * torch.rsqrt(tr.pow(2).mean() + 1e-6) * T16["fnw"]
                at16.k_cache.zero_()
                at16.v_cache.zero_()
                at16.k_cache[:, :, :CAP].copy_(pre[0])
                at16.v_cache[:, :, :CAP].copy_(pre[1])
                yoff = src(hid[q:q + 1], codes[q:q + 1], sem[q:q + 1], tpos[q:q + 1],
                           None if unseen is None else unseen[q:q + 1].to(hid.dtype))[0].float()
                r = {"tower": j, "row": q, "sem": bool(sem[q]), "f32": snr_db(yref, y32), "kv_f32": kv32, "bf16": snr_db(yref, y16),
                     "official_bf16": snr_db(yref, yoff)}
                rows.append(r)
                print(r, flush=True)
                del T32, at32
                gc.collect()
                torch.cuda.empty_cache()
    bl = [r for r in rows if "layer" in r]
    tl = [r for r in rows if "tower" in r]
    summary = {"rows": R, "backbone_f32_min": min(r["f32"] for r in bl), "tower_f32_min": min(r["f32"] for r in tl),
               "tower_semantic_f32_min": min((r["f32"] for r in tl if r["sem"]), default=None),
               "tower_semantic_rows": sum(r["sem"] for r in tl), "backbone_bf16_min": min(r["bf16"] for r in bl), "backbone_official_bf16_min": min(r["official_bf16"] for r in bl)}
    print(summary, flush=True)
    return {"summary": summary, "layers": rows}


# ---------------------------------------------------------------- teacher


def chunk_prompt(fi, model, text: str, ref) -> torch.Tensor:
    """generate_long_drama3 が最初の区切りで generate_drama3 に渡すプロンプト。"""

    class Stop(Exception):
        pass

    got = {}
    orig = fi.generate_drama3

    def grab(model_, prompt, **kw):
        got["p"] = prompt.clone()
        raise Stop

    fi.generate_drama3 = grab
    try:
        for _ in fi.generate_long_drama3(model=model, text=text, max_new_tokens=1024, top_p=0.9, top_k=30, temperature=1.0,
                                         prompt_text=ref[0], prompt_tokens=ref[1]):
            pass
    except Stop:
        pass
    finally:
        fi.generate_drama3 = orig
    return got["p"]


def logit_stats(o: torch.Tensor, e: torch.Tensor) -> dict:
    o, e = o.double(), e.double()
    per = 10 * torch.log10((o ** 2).sum(-1) / ((o - e) ** 2).sum(-1).clamp_min(1e-30))
    lp, lq = torch.log_softmax(o, -1), torch.log_softmax(e, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    return {"snr_db": snr_db(o, e), "snr_min_db": float(per.min()), "top1": float((o.argmax(-1) == e.argmax(-1)).double().mean()),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}


def check_teacher(args, model, ref) -> dict:
    import fish_speech.models.text2semantic.fish_omni.tower as towmod
    from fish_speech.models.text2semantic import inference as fi

    if not args.teacher:
        raise SystemExit("teacher には --teacher（bench.py --save-dir の <ディレクトリ>/<名前>）がいる")
    SEM, EOS = K.SEM_BEGIN, K.EOS
    bb, tw = model.backbone, model.audio_decoder
    eng = K.Drama3Engine(model, wd=args.wd, graphs=False)  # 公式のキャッシュもここで用意される

    def teacher_rows(i):
        c = torch.from_numpy(np.load(REPO / "runs" / f"{args.teacher}_s{args.teacher_seed}_{i:02d}.npy")).long()  # [10, T]
        return torch.cat([(c[0] + SEM)[None], c]).T.contiguous()  # [T, 11]

    @torch.inference_mode()
    def official(prompt, teacher):
        rows, T = len(prompt), len(teacher)
        frames = torch.zeros((rows + T + 1, K.WIDTH), dtype=torch.long, device=DEV)
        frames[:rows] = prompt.to(DEV)
        positions = torch.arange(len(frames), device=DEV)
        for start in range(0, rows, 128):
            end = min(start + 128, rows)
            chunk = frames[start:end]
            emb = tw.embed_backbone(chunk, bb.embed(chunk[:, 0]))
            hidden = bb(inputs_embeds=emb[None], input_pos=positions[start:end])[0].to(DEV)
            pe = min(end, rows - 1)
            if pe > start:
                tw.prefill(hidden[: pe - start], positions[start:pe], frames[start:pe], frames[start + 1: pe + 1])
        hidden = hidden[-1:]
        rm, rc, calls, cur = [], [], [0], [None]
        orig = towmod.sample_top_k_top_p

        def fake(logits, temperature, top_k, top_p, *, check_finite=True):
            # decode_step は 1 フレームに 11 回呼ぶ（main、RAS の引き直し、9 個のコード）。教師の値を返す
            c = calls[0] % 11
            calls[0] += 1
            t = cur[0]
            if c == 0:
                rm.append(logits.float()[0].clone())
            if c <= 1:
                return torch.tensor([0 if t[0] == EOS else t[0] - SEM + 1], device=logits.device)
            rc.append(logits.float()[0].clone())
            return torch.tensor([t[c]], device=logits.device)

        towmod.sample_top_k_top_p = fake
        try:
            for step in range(T):
                p = rows - 1 + step
                cur[0] = teacher[step].tolist()
                logits = bb.compute_logits(hidden).to(DEV)
                s = tw.decode_step(logits, hidden, positions[p:p + 1], frames[p:p + 1], frames[:p + 1, 0], temperature=1.0,
                                   top_k=30, top_p=0.9, epsilon_cutoff=0.0)
                assert s[0].tolist() == cur[0], (s[0].tolist(), cur[0])
                frames[p + 1] = s[0]
                emb = tw.embed_backbone(s, bb.embed(s[:, 0]))
                hidden = bb(inputs_embeds=emb[None], input_pos=positions[p + 1:p + 2])[0].to(DEV)
        finally:
            towmod.sample_top_k_top_p = orig
        return torch.stack(rm), torch.stack(rc).view(T, 9, -1)

    @torch.inference_mode()
    def engine(prompt, teacher):
        T = len(teacher)
        eng.start(eng.prefill(prompt), prompt, 1.0, 30, 0.9)
        em, ec, cur = [], [], [None]
        tdev = teacher.to(DEV)

        def hook(kind, i):  # 引いた値を教師の値で上書きする
            t = cur[0]
            if kind == "main":
                em.append(eng.B.lmain.clone())
                eng.frame_out[0:2].copy_(t[0:2])
                eng.history[15:16].copy_(t[0:1])
            else:
                ec.append(eng.B.lcode.clone())
                eng.frame_out[2 + i:3 + i].copy_(t[2 + i:3 + i])

        for step in range(T):
            cur[0] = tdev[step]
            eng.frame(hook)
        torch.cuda.synchronize()
        assert int(eng.step.item()) == T
        # frames の最後のコードは hook で書き換える前の値なので比べない。前の 10 個は教師と同じはず
        assert torch.equal(eng.frames[:T, :10].cpu(), teacher[:, :10])
        return torch.stack(em), torch.stack(ec).view(T, 9, -1)

    out = {}
    for i in args.idx:
        prompt = chunk_prompt(fi, model, SENTENCES[i]["text"], ref)
        teacher = teacher_rows(i)
        om, oc = official(prompt, teacher)
        bb.residual_dtype = torch.float32  # 公式の fp16 のときの経路（残差を float32 で持つ）を、精度の目安に使う
        try:
            fm, fc = official(prompt, teacher)
        finally:
            bb.residual_dtype = torch.bfloat16
        km, kc = engine(prompt, teacher)
        r = {"rows": len(prompt), "frames": len(teacher),
             "engine_vs_official": {"main": logit_stats(om, km), "codes": [logit_stats(oc[:, j], kc[:, j]) for j in range(9)]},
             "f32res_vs_official": {"main": logit_stats(om, fm), "codes": [logit_stats(oc[:, j], fc[:, j]) for j in range(9)]}}
        out[i] = r
        for nm in ("engine_vs_official", "f32res_vs_official"):
            m, cs = r[nm]["main"], r[nm]["codes"]
            print(f"{i:2d} rows {len(prompt)} T {len(teacher)} {nm:18s} main {m['snr_db']:.1f} dB (min {m['snr_min_db']:.1f}) "
                  f"top1 {m['top1']:.3f} KL {m['kl_mean']:.2e}/{m['kl_max']:.2e} | codes {min(c['snr_db'] for c in cs):.1f}.."
                  f"{max(c['snr_db'] for c in cs):.1f} dB KL max {max(c['kl_max'] for c in cs):.2e}", flush=True)
    del eng
    return {"teacher": args.teacher, "seed": args.teacher_seed, "sentences": out}


# ---------------------------------------------------------------- sampler


def check_sampler(args, model, ref) -> dict:
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

    N = args.draws
    eng = K.Drama3Engine(model, wd=args.wd, graphs=False)
    k, B = eng.k, eng.B
    st = k.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    try:
        from scipy.stats import chi2 as _chi2

        def pval(x, df):
            return float(_chi2.sf(x, df))
    except ImportError:
        def pval(x, df):  # Wilson–Hilferty
            z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
            return 0.5 * math.erfc(z / math.sqrt(2))

    def keep_probs(lg, temp, top_k, top_p):
        """公式の sample_top_k_top_p の確率（float64）。"""
        v = np.asarray(lg, np.float64) / max(temp, 1e-5)
        order = np.argsort(-v, kind="stable")
        sv = v[order]
        p = np.exp(sv - sv.max())
        p /= p.sum()
        drop = (np.cumsum(p) > top_p) | (np.arange(len(v)) >= top_k)
        drop[0] = False
        q = np.where(drop, 0.0, p)
        q /= q.sum()
        o = np.zeros(len(v))
        o[order] = q
        return o

    results = []

    def test(name, draws, expect):
        cnt = np.bincount(draws, minlength=len(expect)).astype(np.float64)
        kept = expect > 0
        outside = int(cnt[~kept].sum())
        e = expect[kept] * len(draws)
        x = float(((cnt[kept] - e) ** 2 / e).sum())
        df = max(int(kept.sum()) - 1, 1)
        r = {"name": name, "kept": int(kept.sum()), "chi2": x, "df": df, "p": pval(x, df),
             "tv": 0.5 * float(np.abs(cnt / len(draws) - expect).sum()), "outside": outside}
        results.append(r)
        print(f"{name:44s} kept {r['kept']:3d}  chi2 {x:8.1f} df {df:3d} p {r['p']:.3f}  TV {r['tv']:.4f}  outside {outside}",
              flush=True)

    def run_code(logits, temp, top_k, top_p):
        """sample_code（最後のコード）を N 回。step と counter は毎回 1 つ進む。"""
        B.lcode.copy_(torch.as_tensor(logits, dtype=torch.float32, device=DEV))
        eng.params.copy_(torch.tensor([temp, top_p, float(top_k)]))
        res = []
        with torch.inference_mode(), st:
            done = 0
            while done < N:
                n = min(N - done, eng.cap)
                eng.step.zero_()
                for _ in range(n):
                    k("sample_code", 1, 1024, (B.lcode, eng.params, eng.seed, eng.counter, eng.frame_out, np.int32(10),
                                               np.int32(1), eng.frames, eng.step, eng.bad))
                res.append(eng.frames[:n, 10].clone())
                done += n
        torch.cuda.synchronize()
        return torch.cat(res).cpu().numpy()

    def run_main(logits, temp, top_k, top_p, history):
        B.lmain.copy_(torch.as_tensor(logits, dtype=torch.float32, device=DEV))
        eng.params.copy_(torch.tensor([temp, top_p, float(top_k)]))
        h0 = torch.as_tensor(history, dtype=torch.int64, device=DEV)
        o = torch.zeros(N, dtype=torch.int64, device=DEV)
        with torch.inference_mode(), st:
            eng.step.zero_()
            for j in range(N):
                eng.history.copy_(h0)
                k("sample_main", 1, 1024, (B.lmain, np.int32(4097), eng.params, eng.seed, eng.counter, eng.history,
                                           eng.frame_out, eng.host_main, eng.step, eng.bad, np.int32(K.EOS),
                                           np.int32(K.SEM_BEGIN), np.int32(K.SEM_END)))
                o[j:j + 1].copy_(eng.frame_out[0:1])
                eng.counter.add_(1)
        torch.cuda.synchronize()
        m = o.cpu().numpy()
        return np.where(m == K.EOS, 0, m - K.SEM_BEGIN + 1)  # 候補の番号（0 = eos）

    eng.seed.fill_(12345)
    eng.counter.zero_()
    eng.bad.zero_()
    rng = np.random.default_rng(0)
    L = 2.5 * rng.standard_normal(4096)
    for temp, tk, tp in ((1.0, 30, 0.9), (0.7, 30, 0.7), (1.0, 64, 1.0), (1.3, 10, 0.95), (1.0, 30, 1e-4)):
        test(f"code T={temp} k={tk} p={tp}", run_code(L, temp, tk, tp), keep_probs(L, temp, tk, tp))
    flat = 0.05 * rng.standard_normal(4096)  # 平らなロジット（top-k が効く）
    test("code flat T=1 k=30 p=0.9", run_code(flat, 1.0, 30, 0.9), keep_probs(flat, 1.0, 30, 0.9))
    peak = L.copy()  # 尖ったロジット（top-p が効く）
    peak[7] += 8.0
    test("code peaked T=1 k=30 p=0.9", run_code(peak, 1.0, 30, 0.9), keep_probs(peak, 1.0, 30, 0.9))

    # main token: 最初の入力のあとの実際のロジット
    prompt = build_drama3_prompt(LAYER_TEXT, model.tokenizer, model.config)
    with torch.inference_mode():
        eng.start(eng.prefill(prompt), prompt, 1.0, 30, 0.9)
        with st:
            k.gemv(B.hid, None, 0.0, None, eng.lm)
    torch.cuda.synchronize()
    lm = B.lmain.double().cpu().numpy()
    P1 = keep_probs(lm, 1.0, 30, 0.9)
    P2 = keep_probs(lm, 1.0, 30, 0.95)  # RAS の引き直し（温度 1.0、top-k 30、top-p 0.95）
    test("main T=1 k=30 p=0.9, history empty", run_main(lm, 1.0, 30, 0.9, [-1] * 16), P1)
    top = np.argsort(-P1)[:3]
    hist = [-1] * 13 + [int(K.SEM_BEGIN + t - 1) for t in top]
    rep = np.zeros(len(P1), bool)
    rep[top] = True
    rep[0] = False  # eos は引き直さない
    test(f"main RAS: history has top-3 {[int(t) for t in top]}", run_main(lm, 1.0, 30, 0.9, hist),
         np.where(rep, 0.0, P1) + P1[rep].sum() * P2)
    # eos を上位にして、履歴に eos を入れる（eos は semantic でないので引き直さない）
    lm2 = lm.copy()
    lm2[0] = lm.max() + 0.5
    P1e, P2e = keep_probs(lm2, 1.0, 30, 0.9), keep_probs(lm2, 1.0, 30, 0.95)
    top2 = np.argsort(-P1e)[:3]
    hist2 = [-1] * 12 + [K.EOS] + [int(K.SEM_BEGIN + t - 1) if t else K.EOS for t in top2]
    rep2 = np.zeros(len(P1e), bool)
    rep2[[t for t in top2 if t != 0]] = True
    test(f"main RAS with eos in history, top {[int(t) for t in top2]}", run_main(lm2, 1.0, 30, 0.9, hist2),
         np.where(rep2, 0.0, P1e) + P1e[rep2].sum() * P2e)
    bad = int(eng.bad.item())
    print("bad flag", bad, flush=True)
    del eng
    return {"draws": N, "bad_flag": bad, "tests": results}


# ---------------------------------------------------------------- prefill


def check_prefill(args, model, ref) -> dict:
    from fish_speech.models.text2semantic import inference as fi
    from fish_speech.models.text2semantic.inference import Drama3History

    if not ref[0]:
        raise SystemExit("prefill には --ref と --ref-text がいる（参照の使い回しを確かめるため）")
    bb = model.backbone
    eng = K.Drama3Engine(model, wd=args.wd, graphs=True)

    def P(i, use_ref=True):
        return fi.build_drama3_chunk_prompt(model, SENTENCES[i]["text"], ref[0] if use_ref else [],
                                            ref[1] if use_ref else [], Drama3History())

    def snapshot(rows):
        d = {"hidden": eng.pf_hidden.clone()}
        for i, layer in enumerate(bb.model.layers):
            if layer.layer_type == "full_attention":
                a = layer.self_attn
                d[f"k{i}"], d[f"v{i}"] = a.k_cache[0, :, :rows].clone(), a.v_cache[0, :, :rows].clone()
            else:
                g = layer.linear_attn
                d[f"s{i}"], d[f"c{i}"] = g.recurrent_state[0].clone(), g.conv_state[0].clone()
        for j, a in enumerate(eng._tower_attentions()):
            d[f"tk{j}"], d[f"tv{j}"] = a.k_cache[0, :, :rows - 1].clone(), a.v_cache[0, :, :rows - 1].clone()
        return d

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    results = []

    def report(name, base, got, sec):
        v = {k_: snr_db(base[k_], got[k_]) for k_ in base}
        worst = min(v, key=v.get)
        same = all(torch.equal(base[k_], got[k_]) for k_ in base)
        r = {"name": name, "ms": sec * 1e3, "bit_identical": same, "hidden_db": v["hidden"], "min_db": v[worst],
             "min_at": worst, "gdn_state_min_db": min(x for k_, x in v.items() if k_.startswith("s"))}
        results.append(r)
        print(f"{name:34s} {sec * 1e3:6.1f} ms  identical {same}  hidden {v['hidden']:.1f} dB  min {v[worst]:.1f} dB "
              f"({worst})  GDN state min {r['gdn_state_min_db']:.1f} dB", flush=True)

    def eager_reset():
        eng.pf_capacity, eng.prefix_cache = 0, False
        eng.prefix = eng.last_prompt = None

    with torch.inference_mode():
        # 1. グラフ: 同じ長さの別のプロンプト（A で記録、B で流す）を、eager の B と比べる
        by = {}
        for i in range(len(SENTENCES)):
            by.setdefault(len(P(i, False)), []).append(i)
        rows_, idx = max(((r, v) for r, v in by.items() if len(v) >= 2), key=lambda x: len(x[1]))
        A, Bp = P(idx[0], False), P(idx[1], False)
        assert not torch.equal(A, Bp)
        eager_reset()
        eng.prefill(Bp)  # 1 回目は Triton の読み込みなどが入るので、2 回目を測る
        te = timed(lambda: eng.prefill(Bp))
        base = snapshot(rows_)
        eng.pf_capacity = 16
        eng.prefill(A)  # 初めての区切りは eager
        eng.prefill(A)  # 2 回目で (0, rows) のグラフを記録する
        assert eng.pf_captures >= 1 and len(eng.pf_graphs) >= 1
        tg = timed(lambda: eng.prefill(Bp))
        report(f"graph, no ref, rows {rows_} (eager {te * 1e3:.0f} ms)", base, snapshot(rows_), tg)
        # 2. 参照の使い回し: q1, q2（ここで状態を写す）, q3, q4（使い回す）を、eager で q3・q4 だけを流したときと比べる
        q1, q2, q3, q4 = P(3), P(8), P(12), P(20)
        base = {}
        for name, q in (("q3", q3), ("q4", q4)):
            eager_reset()
            eng.prefill(q)
            t = timed(lambda: eng.prefill(q))
            base[name] = snapshot(len(q))
            print(f"eager {name}: rows {len(q)}, {t * 1e3:.1f} ms", flush=True)
            # 公式どうしの差（区切りを 64 行にしたとき、残差を float32 にしたとき）を、使い回しの差の目安にする
            eng.prefill(q, chunk=64)
            report(f"official chunk 64 vs 128, {name}", base[name], snapshot(len(q)), 0.0)
            bb.residual_dtype = torch.float32
            try:
                eng.prefill(q)
                report(f"official fp32 residual vs bf16, {name}", base[name], snapshot(len(q)), 0.0)
            finally:
                bb.residual_dtype = torch.bfloat16
        eager_reset()
        eng.prefix_cache = True
        for q in (q1, q2):
            eng.prefill(q)
        common = None if eng.prefix is None else len(eng.prefix)
        print(f"common rows {K._common_rows(q1, q2)}, stored prefix {common}", flush=True)
        t = timed(lambda: eng.prefill(q3))
        report("prefix, eager, q3", base["q3"], snapshot(len(q3)), t)
        t = timed(lambda: eng.prefill(q4))
        report("prefix, eager, q4", base["q4"], snapshot(len(q4)), t)
        # 3. 参照の使い回し + グラフ（区切りは 2 回目に記録するので、2 回流してから 3 回目を測る）
        eng.pf_capacity = 16
        eng.prefix = eng.last_prompt = None
        for q in (q1, q2, q3, q3):
            eng.prefill(q)
        t = timed(lambda: eng.prefill(q3))
        report("prefix + graph, q3", base["q3"], snapshot(len(q3)), t)
        eng.prefill(q4)
        eng.prefill(q4)
        t = timed(lambda: eng.prefill(q4))
        report("prefix + graph, q4", base["q4"], snapshot(len(q4)), t)
        # 4. グラフと eager が混ざるとき（使い回しなし。(0, 128) は記録ずみのグラフ、残りの区切りは初めてなので eager）
        eng.prefix_cache, eng.prefix, eng.last_prompt = False, None, None
        tail = (128, len(q4), len(q4) - 1)
        assert (0, 128, 128) in eng.pf_graphs and tail not in eng.pf_graphs and eng.pf_seen.get(tail, 0) == 0
        t = timed(lambda: eng.prefill(q4))
        report("graph + eager mixed, q4", base["q4"], snapshot(len(q4)), t)
    out = {"common_rows": K._common_rows(q1, q2), "graphs": len(eng.pf_graphs), "captures": eng.pf_captures, "tests": results}
    del eng
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("layers", "teacher", "sampler", "prefill"))
    ap.add_argument("--wd", choices=("bf16", "i8g32"), default="bf16", help="teacher・sampler・prefill のエンジンの重み")
    ap.add_argument("--ref", default="", help="参照の WAV（runs/ からの相対、または絶対のパス）")
    ap.add_argument("--ref-text", default="", help="参照の書き起こし")
    ap.add_argument("--teacher", default="", help="teacher: 公式の出力の .npy（runs/ の下の <ディレクトリ>/<名前>）")
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--idx", default="0,5,17,30", help="teacher: 文の番号")
    ap.add_argument("--draws", type=int, default=40000, help="sampler: 1 通りあたりに引く回数")
    args = ap.parse_args()
    args.idx = [int(x) for x in args.idx.split(",")]
    from fish_speech.models.text2semantic import inference as fi

    path = checkpoint_path()
    model, _ = fi.init_model(str(path), "cuda:0", torch.bfloat16)
    ref: tuple[list, list] = ([], [])
    info: dict = {"what": args.what, "wd": args.wd, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if args.ref:
        if not args.ref_text:
            raise SystemExit("--ref には --ref-text がいる")
        codec = fi.load_codec_model(str(path / "codec.pth"), "cuda:0", torch.bfloat16)
        codes, sec = encode_reference(codec, args.ref)
        ref = ([args.ref_text], [codes])
        info["reference"] = {"file": args.ref, "seconds": sec, "frames": int(codes.shape[-1])}
        del codec
        gc.collect()
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    fn = {"layers": check_layers, "teacher": check_teacher, "sampler": check_sampler, "prefill": check_prefill}[args.what]
    info["result"] = fn(args, model, ref)
    info["sec"] = time.perf_counter() - t0
    suffix = ("" if args.what == "layers" else f"-{args.wd}") + ("-ref" if args.ref else "")
    out = REPO / "runs" / "check" / "drama3" / f"{args.what}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"saved {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
