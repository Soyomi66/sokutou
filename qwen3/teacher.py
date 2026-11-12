"""教師強制（決まったコードの列を入れる）で、各ステップの logits を float32 の基準と比べる。

bfloat16 ではサンプリングが数フレームで公式と分かれるので、コードが同じかどうかでは確かめられない。
代わりに、公式の実装の出力のコード（runs/bench/qwen3/official-<model>-wav/*.npy）を入力として与え、
フレームごとに、言語モデルの第 0 のコードの logits（1 + フレーム数 個）と、code predictor の logits（フレーム × 15 個）を
集め、float32 の基準と比べる。

  # 基準（float32、公式と同じ計算の経路）を保存する
  ... scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant hf-dynamic --dtype float32 --save-ref
  # 比べる（公式と同じ経路の bfloat16 が、比べるときの下限の目安）
  ... scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant hf-dynamic --dtype bfloat16
  ... scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant hf-static --dtype bfloat16

variant
  hf-dynamic : 公式と同じ（DynamicCache、マスクなしの SDPA、code predictor はフレームごとに新しいキャッシュ）
  hf-static  : qwen3/fast.py と同じ（StaticCache と、キャッシュの長さのマスク）
指標（ステップごとに出して平均する）
  kl      : 基準の分布から見た KL（softmax(logits / 0.9)。公式のサンプリングの温度）
  top1    : いちばん大きい logits の番号が基準と同じ割合
  rel     : logits の差のノルム / 基準の logits のノルム
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache, StaticCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import LANGUAGE, SPEAKER, load  # noqa: E402
from fast import Qwen3Fast  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEMP = 0.9


@torch.no_grad()
def teacher_hf(tts, kw: dict, codes: torch.Tensor, static: bool, max_cache: int = 1024):
    """HF のモジュールで、codes（frames, 16）を入れたときの logits を返す（float32、CPU）。"""
    talker = tts.model.talker
    cp = talker.code_predictor
    dev = talker.device
    embeds = kw["inputs_embeds"]
    pad = kw["tts_pad_embed"]
    L = embeds.shape[1]
    if static:
        tcache = StaticCache(config=talker.config, max_cache_len=max_cache)
        ccache = StaticCache(config=cp.config, max_cache_len=16)
        out = talker.model(inputs_embeds=embeds, past_key_values=tcache, cache_position=torch.arange(L, device=dev),
                           use_cache=True)
    else:
        tcache = DynamicCache()
        out = talker.model(inputs_embeds=embeds, attention_mask=torch.ones((1, L), dtype=torch.long, device=dev),
                           past_key_values=tcache, use_cache=True)
    hidden = out.last_hidden_state
    l0 = [talker.codec_head(hidden)[:, -1, :].float().cpu()]
    lcp = []
    cp_emb = cp.get_input_embeddings()
    for f in range(codes.shape[0]):
        c = codes[f].view(1, 16).to(dev)
        past = hidden[:, -1:, :]
        emb0 = talker.get_input_embeddings()(c[:, :1])
        if not static:
            ccache = DynamicCache()
        x = cp.small_to_mtp_projection(torch.cat([past, emb0], dim=1))
        kwc = {"cache_position": torch.arange(2, device=dev)} if static else {}
        o = cp.model(inputs_embeds=x, past_key_values=ccache, use_cache=True, **kwc)
        frame = [cp.lm_head[0](o.last_hidden_state)[:, -1, :].float().cpu()]
        embs = [emb0, cp_emb[0](c[:, 1:2])]
        for g in range(1, 15):
            e = cp.small_to_mtp_projection(cp_emb[g - 1](c[:, g:g + 1]))
            kwc = {"cache_position": torch.tensor([g + 1], device=dev)} if static else {}
            o = cp.model(inputs_embeds=e, past_key_values=ccache, use_cache=True, **kwc)
            frame.append(cp.lm_head[g](o.last_hidden_state)[:, -1, :].float().cpu())
            embs.append(cp_emb[g](c[:, g + 1:g + 2]))
        lcp.append(torch.cat(frame, dim=0))
        x = torch.cat(embs, dim=1).sum(1, keepdim=True) + pad
        if static:
            out = talker.model(inputs_embeds=x, past_key_values=tcache,
                               cache_position=torch.tensor([L + f], device=dev), use_cache=True)
        else:
            out = talker.model(inputs_embeds=x, attention_mask=torch.ones((1, L + f + 1), dtype=torch.long, device=dev),
                               past_key_values=tcache, use_cache=True)
        hidden = out.last_hidden_state
        l0.append(talker.codec_head(hidden)[:, -1, :].float().cpu())
    return torch.cat(l0, dim=0), torch.stack(lcp, dim=0)


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """ref, got は (..., V)。最後の軸ごとに比べて平均する。"""
    ref = ref.reshape(-1, ref.shape[-1]).double()
    got = got.reshape(-1, got.shape[-1]).double()
    lp = torch.log_softmax(ref / TEMP, -1)
    lq = torch.log_softmax(got / TEMP, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (ref.argmax(-1) == got.argmax(-1)).double()
    rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(top1.mean()),
            "rel_mean": float(rel.mean()), "n": int(ref.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--variant", choices=["hf-dynamic", "hf-static", "kernels"], default="hf-dynamic")
    ap.add_argument("--wd", default="bf16", help="--variant kernels の重みの型（bf16、i8g32、f32）")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--save-ref", action="store_true", help="基準として runs/check/qwen3/teacher-ref-<model>.pt に保存する")
    args = ap.parse_args()
    tts = load(args.model, args.dtype, "sdpa")
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))[: args.limit]
    wav_dir = REPO / "runs" / "bench" / "qwen3" / f"official-{args.model}-wav"
    probe = Qwen3Fast(tts, graphs=False)
    if args.variant == "kernels":
        from kernels import Qwen3Kernels

        eng = Qwen3Kernels(tts, wd=args.wd, graphs=False)
        eng.capture()  # StaticCache を用意する（記録したグラフは使わない）
    res = []
    for i, s in enumerate(sentences):
        codes = torch.from_numpy(np.load(wav_dir / f"official-{args.model}-1_s1_{i:02d}.npy")).long()
        kw = probe.prompt(s["text"], SPEAKER, LANGUAGE)
        if args.variant == "kernels":
            res.append(eng.teacher_logits(kw, codes))
        else:
            res.append(teacher_hf(tts, kw, codes, static=args.variant == "hf-static"))
    ref_path = REPO / "runs" / "check" / "qwen3" / f"teacher-ref-{args.model}.pt"
    if args.save_ref:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, ref_path)
        print("saved", ref_path, [tuple(r[1].shape) for r in res])
        return
    ref = torch.load(ref_path)
    m0 = metrics(torch.cat([r[0] for r in ref]), torch.cat([g[0] for g in res]))
    mc = metrics(torch.cat([r[1] for r in ref]), torch.cat([g[1] for g in res]))
    out = {"model": args.model, "variant": args.variant + (f"-{args.wd}" if args.variant == "kernels" else ""),
           "dtype": args.dtype, "talker": m0, "code_predictor": mc}
    print(json.dumps(out, ensure_ascii=False))
    with (REPO / "runs" / "check" / "qwen3" / "teacher.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
