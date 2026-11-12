"""エンジン（engine/fast.py）を保存した参照（runs/ref/<tag>/）と同じ入力で動かし、tools/snr.py で比べられる形で書く。

書くもの（runs/cand/<name>/<case>/）:
- cond.npz     : エンジンが計算した文の状態（有効なトークンだけ）と話者の状態
- kv.npz       : 参照の条件から計算した K・V（層 0 と 11）
- steps_tf.npz : ステップごとに参照の x と参照の条件を入れたときの v（グラフに記録するのと同じ step 関数）
- blocks_s00/20.npz : 同じく各層の出力
- audio_tf.npz : 参照の潜在変数の復号と、参照の音声への電子透かし（公式の処理。エンジンを動かした後に同じプロセスで呼ぶ）
- free.npz     : 保存したノイズから CUDA Graph で通した結果（最後の x、復号、電子透かし、24kHz の出力）

あわせて runs/checks/<name>.json に、TF32 が DiT の外に漏れていないことの確かめを書く:
- 公式の encode_conditions（パディングあり）の文と話者の状態、復号、電子透かしが、参照と一致するか（最大絶対誤差）
- 行列積の試し（今のフラグのまま、無効、有効で計算して比べる）
- グラフの再生と、グラフなしの実行の差

  scripts/run_in_image.sh engine/check.py --precision fp32 --name fast-fp32
  scripts/run_in_image.sh engine/check.py --precision tf32 --name fast-tf32
  scripts/run_in_image.sh engine/check.py --precision tf32 --name fast-tf32-global --global-tf32   # 対照: 全体に効かせる
  python3 tools/snr.py runs/ref/gpu-s20 runs/cand/fast-fp32

ステップ数の既定は 20（ref/common.py の DEFAULT_STEPS）で、参照も同じステップ数のもの（既定は runs/ref/gpu-s20）を使う。
40 ステップで確かめるときは --steps 40 を付ける（参照は runs/ref/gpu）。
MF 版は IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF で選ぶ（既定は 4 ステップ、参照は runs/ref/gpu-mf）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
sys.path.insert(0, "/work")
import irodori_tts.inference_runtime as IR  # noqa: E402
from common import DEFAULT_STEPS, WORK, load_runtime, make_request, postprocess, set_num_steps, steps_tag  # noqa: E402
from dump_reference import BLOCK_TRACE_STEPS, np32, stack  # noqa: E402
from engine.fast import FastSampler, matmul_tf32, runtime_sampler_name  # noqa: E402


def load(p: Path) -> dict[str, np.ndarray]:
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def maxabs(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)))


def matmul_probe(device) -> dict:
    """今のフラグのままの行列積が、TF32 無効の結果と一致し、有効の結果と違うことを見る。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(270, 1280, generator=g).to(device)
    b = torch.randn(1280, 1280, generator=g).to(device)
    as_is = a @ b
    with matmul_tf32(False):
        off = a @ b
    with matmul_tf32(True):
        on = a @ b
    return {"flag": bool(torch.backends.cuda.matmul.allow_tf32),
            "as_is_vs_off_maxabs": float((as_is - off).abs().max()),
            "as_is_vs_on_maxabs": float((as_is - on).abs().max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default=None, help="runs/ref/ の下の参照。既定は --steps に合わせる（40 なら gpu、20 なら gpu-s20、MF 版の 4 なら gpu-mf）")
    ap.add_argument("--global-tf32", action="store_true",
                    help="対照: エンジンを作った後、allow_tf32 をプロセス全体で有効にする（漏れの確かめが見分けられるかを見る）")
    ap.add_argument("--no-graph-text", action="store_true", help="文のエンコードをグラフの外で行う")
    ap.add_argument("--engine", default="exact", choices=("exact", "bucket", "fused"),
                    help="exact: 形ごとに全ステップを記録する（engine/fast.py）。bucket: 詰め物とステップごとのグラフ（engine/bucket.py）。"
                         "fused: bucket の小さな処理を自作のカーネルにまとめたもの（engine/fused.py）")
    ap.add_argument("--bucket-decode", action="store_true",
                    help="通しの比較で、復号を決まった長さで動かす（engine/decode.py）。公式の復号の確かめ（4.）は元の復号で行う")
    ap.add_argument("--latent-buckets", default="", help="--engine bucket の潜在の長さの区切り（カンマ区切り。既定は engine/bucket.py）")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"サンプリングのステップ数（既定 {DEFAULT_STEPS}。ref/common.py の DEFAULT_STEPS）。同じステップ数で保存した参照を使う")
    ap.add_argument("--fp8", nargs="?", const="all", default=None, choices=("all", "cfg", "nocfg"), help="--engine fused で、DiT の行列積を FP8 にする（engine/fp8.py。fp32 の関門は通らない）")
    args = ap.parse_args()
    if args.ref is None:
        args.ref = "gpu" + steps_tag(args.steps)
    set_num_steps(args.steps)
    rt = load_runtime()
    model = rt.model
    dev = model.device
    bucket = args.engine in ("bucket", "fused")
    skw = {"num_steps": args.steps, **({"fp8": args.fp8} if args.fp8 else {})}
    if bucket:
        from engine.bucket import BucketSampler
        if args.engine == "fused":
            from engine.fused import FusedBucketSampler as BucketSampler
        bkw = {"latent_buckets": [int(x) for x in args.latent_buckets.split(",")]} if args.latent_buckets else {}
        sampler = BucketSampler(model, precision=args.precision, **bkw, **skw)
        eager = BucketSampler(model, precision=args.precision, use_graph=False, **bkw, **skw)
    else:
        sampler = FastSampler(model, precision=args.precision, graph_text=not args.no_graph_text, **skw)
        eager = FastSampler(model, precision=args.precision, graph_text=not args.no_graph_text, use_graph=False, **skw)
    if args.global_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    sampler_name = runtime_sampler_name(model)  # RF 版は sample_euler_rf_cfg、MF 版は sample_euler_meanflow
    official_sample = getattr(IR, sampler_name)
    got: dict = {}
    noise: dict = {}

    def sample(**kw):
        kw["initial_noise"] = noise["x"]
        got["kw"] = kw
        z = sampler.sample(**kw)
        got["z"] = np32(z)
        got["text_state"] = np32(sampler.last_text_state)
        return z

    dec, wm = rt.codec.decode_latent, rt.watermarker.encode_batch
    bdec = None
    if args.bucket_decode:
        from engine.decode import BucketDecoder
        bdec = BucketDecoder(rt)

    def decode(z):
        a = dec(z) if bdec is None else bdec.decode_latent(z)
        got["decode_out"] = np32(a)
        return a

    def watermark(audios, *, sample_rate):
        out = wm(audios, sample_rate=sample_rate)
        got["wm_out"] = np32(out[0])
        return out

    def T(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a).to(dev)
        return t if t.dtype in (torch.bool, torch.int64) else t.to(model.dtype)

    ref_root = WORK / "runs" / "ref" / args.ref
    out_root = WORK / "runs" / "cand" / args.name
    checks: dict = {"precision": args.precision, "global_tf32": args.global_tf32, "cases": {}}
    for ref_dir in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        meta = json.loads((ref_dir / "meta.json").read_text())
        assert int(meta["num_steps"]) == args.steps, f"参照は {meta['num_steps']} ステップ。--steps と --ref を合わせる"
        inp, rcond, rsteps, raud = (load(ref_dir / f) for f in ("inputs.npz", "cond.npz", "steps.npz", "audio.npz"))
        out_dir = out_root / ref_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        c: dict = {}

        # 1. 通し: 保存したノイズを入れて、公式の synthesize の流れ（復号、電子透かし、後処理は公式）で動かす。
        setattr(IR, sampler_name, sample)
        rt.codec.decode_latent = decode
        rt.watermarker.encode_batch = watermark
        noise["x"] = T(inp["noise"])
        req = make_request(meta["text"], meta["seconds"], meta["seed"])
        rt.synthesize(req, log_fn=None)  # グラフを記録する
        got.clear()
        out24k = postprocess(rt.synthesize(req, log_fn=None))  # 再生
        assert sampler.fallbacks == 0, "公式の関数に落ちた"
        np.savez(out_dir / "free.npz", x_final=got["z"], decode_out=got["decode_out"], wm_out=got["wm_out"],
                 out24k=out24k)
        setattr(IR, sampler_name, official_sample)
        rt.codec.decode_latent = dec
        rt.watermarker.encode_batch = wm
        c["flag_after_replay"] = bool(torch.backends.cuda.matmul.allow_tf32)

        with torch.inference_mode():
            n = int(inp["text_mask"].sum())
            ids = T(inp["text_ids"])[:, :n]
            # グラフなしで同じ入力を通し、グラフの再生と比べる。
            voice = sampler.voice_for(T(inp["ref_latent"]), T(inp["ref_mask"]))
            z_eager = eager.sample(**got["kw"])
            c["graph_vs_eager_x_final_maxabs"] = maxabs(got["z"], np32(z_eager))
            c["x_final_finite"] = bool(np.isfinite(got["z"]).all())

            # 2. エンジンが計算した条件。
            c["text_state_graph_vs_eager_maxabs"] = maxabs(got["text_state"], np32(eager.last_text_state))
            np.savez(out_dir / "cond.npz", text_state=got["text_state"], text_mask=np.ones((1, n), dtype=bool),
                     speaker_state=np32(voice.state), speaker_mask=np.ones((1, voice.tokens), dtype=bool))

            # 3. ステップごとに参照の入力を入れる。条件も参照のもの（有効なトークンだけ）。
            ref_text = T(rcond["text_state"])[:, :n]
            ref_voice = sampler.make_voice(T(rcond["speaker_state"]), T(rcond["speaker_mask"]))
            seq_len = int(inp["noise"].shape[1])
            v_cfg, v_nocfg, v_all = [], [], []
            if bucket:
                sh = sampler.tf_prepare(seq_len, ref_text, ref_voice)
                c["shape"] = {"latent": seq_len, "latent_bucket": sh.lb, "text": n, "text_bucket": sh.tb,
                              "text_encode_bucket": min(b for b in sampler.text_encode_buckets if b >= n)}
                kv_out = {}
                for li in (0, sh.ctx.k.shape[0] - 1):
                    kv_out[f"kv_l{li}_0"] = np32(sh.ctx.k[li][:, :n])
                    kv_out[f"kv_l{li}_1"] = np32(sh.ctx.v[li][:, :n])
                    kv_out[f"kv_l{li}_2"] = np32(sh.ctx.k[li][:, sh.tb:])
                    kv_out[f"kv_l{li}_3"] = np32(sh.ctx.v[li][:, sh.tb:])
                np.savez(out_dir / "kv.npz", **kv_out)
                for i in range(len(rsteps["t"])):
                    v, raw = sampler.tf_step(sh, T(rsteps["x"][i]), i)
                    (v_cfg if sampler.use_cfg[i] else v_nocfg).append(np32(raw))
                    v_all.append(np32(v))
            else:
                mask3 = sampler.cfg_mask(seq_len, n, ref_voice.tokens)
                with matmul_tf32(args.precision == "tf32"):
                    ctx = sampler.build_context(ref_text, ref_voice.kv)
                    kv_out = {}
                    for li in (0, ctx[0].shape[0] - 1):
                        kv_out[f"kv_l{li}_0"] = np32(ctx[0][li][:, :n])
                        kv_out[f"kv_l{li}_1"] = np32(ctx[1][li][:, :n])
                        kv_out[f"kv_l{li}_2"] = np32(ctx[0][li][:, n:])
                        kv_out[f"kv_l{li}_3"] = np32(ctx[1][li][:, n:])
                    np.savez(out_dir / "kv.npz", **kv_out)
                    for i in range(len(rsteps["t"])):
                        x = T(rsteps["x"][i])
                        _, v, raw = sampler.step(x, i, ctx, mask3)
                        (v_cfg if sampler.use_cfg[i] else v_nocfg).append(np32(raw))
                        v_all.append(np32(v))
                        if i in BLOCK_TRACE_STEPS:
                            trace: dict = {}
                            xin = x.expand(3, -1, -1) if sampler.use_cfg[i] else x
                            sampler.forward(xin, i, ctx, mask3 if sampler.use_cfg[i] else None, trace)
                            np.savez(out_dir / f"blocks_s{i:02d}.npz", **{k: np32(t) for k, t in trace.items()})
            assert sampler.use_cfg == [i in set(int(j) for j in rsteps["cfg_steps"]) for i in range(len(rsteps["t"]))]
            np.savez(out_dir / "steps_tf.npz", v_raw_cfg=stack(v_cfg), v_raw_nocfg=stack(v_nocfg),
                     v=np.stack(v_all))

            # 4. エンジンの外の処理が、参照と一致するか（TF32 が漏れていれば、ここが変わる）。
            off = model.encode_conditions(
                text_input_ids=T(inp["text_ids"]), text_mask=T(inp["text_mask"]),
                ref_latent=T(inp["ref_latent"]), ref_mask=T(inp["ref_mask"]),
                caption_input_ids=T(inp["caption_ids"]), caption_mask=T(inp["caption_mask"]),
                speaker_uncond_mode="mask")
            c["official_text_state_maxabs"] = maxabs(rcond["text_state"], np32(off[0]))
            c["official_speaker_state_maxabs"] = maxabs(rcond["speaker_state"], np32(off[2]))
            c["engine_speaker_state_vs_ref_maxabs"] = maxabs(rcond["speaker_state"], np32(voice.state))
            dec_out = rt.codec.decode_latent(T(raud["decode_in"]))
            wm_out = rt.watermarker.encode_batch([torch.from_numpy(raud["wm_in"])], sample_rate=rt.codec.sample_rate)
            np.savez(out_dir / "audio_tf.npz", decode_out=np32(dec_out), wm_out=np32(wm_out[0]))
            c["official_decode_maxabs"] = maxabs(raud["decode_out"], np32(dec_out))
            c["official_watermark_maxabs"] = maxabs(raud["wm_out"], np32(wm_out[0]))
            c["matmul_probe"] = matmul_probe(dev)
        checks["cases"][ref_dir.name] = c
        print("done", ref_dir.name, json.dumps(c), flush=True)
    if bucket:
        checks["late_captures"] = sampler.late_captures
        checks["bucket_graphs"] = sampler.graph_count()
    checks["graphs"] = len(sampler.graphs)
    checks["capture_sec"] = sampler.capture_sec
    out = WORK / "runs" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
