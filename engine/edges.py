"""本番の入力の範囲の端で、バケットのエンジンが落ちず、公式の関数にも戻らないことを確かめる。

  scripts/run_in_image.sh engine/edges.py            # BF16
  IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/edges.py --engine fused --no-watermark
ステップ数の既定は 20（ref/common.py の DEFAULT_STEPS）。40 以外は名前に -s<数> が付く。
出力: runs/checks/edges-<精度>.json（--engine fused は edges-fused-<精度>[-nowm][-hdec].json）（時間は、起動の準備のあとの 1 回目と 2 回目）
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/ref")
from common import DEFAULT_STEPS, WORK, load_runtime, make_request, normalize_text, set_num_steps, steps_tag, timed_synth  # noqa: E402

from engine.bucket import install_bucket, prewarm_codec  # noqa: E402

LONG = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。"
KANJI = "鬱蒼たる叢林に魑魅魍魎が跋扈し、轟々たる驟雨が霹靂と共に襲来する。"


def cases() -> list[tuple[str, str, float | None]]:
    t200 = (LONG * 3)[:200]
    k200 = (KANJI * 8)[:200]
    return [
        ("1 文字、0.5 秒", "あ", 0.5),
        ("短い文、0.5 秒", "はい。", 0.5),
        ("短い文、30 秒", "はい。", 30.0),
        ("200 文字、30 秒", t200, 30.0),
        ("200 文字、0.5 秒", t200, 0.5),
        ("200 文字、秒数の指定なし", t200, None),
        ("難しい漢字 200 文字、30 秒", k200, 30.0),
        ("短い文、秒数の指定なし", "こんにちは。", None),
        ("短い文、0.25 秒（speed 2.0 で 0.5 秒）", "はい。", 0.25),
        ("200 文字、60 秒（speed 0.5 で 30 秒）", t200, 60.0),
        ("英数字と記号 200 文字、30 秒", ("GPU-0 の RTX6000Ada で 48kHz→24kHz, v4.1-Small (40 steps) #test! " * 4)[:200], 30.0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="bf16", choices=("fp32", "tf32", "bf16"))
    ap.add_argument("--engine", default="bucket", choices=("bucket", "fused"))
    ap.add_argument("--no-watermark", action="store_true", help="電子透かしを外す（engine/bucket.py の disable_watermark）")
    ap.add_argument("--half-decode", action="store_true", help="復号を float16 の autocast で動かす（engine/fused.py の half_decode）")
    ap.add_argument("--bucket-decode", action="store_true", help="復号を決まった長さで動かす（engine/decode.py）")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"サンプリングのステップ数（既定 {DEFAULT_STEPS}。ref/common.py の DEFAULT_STEPS）。40 以外は名前に -s<数> が付く")
    ap.add_argument("--fp8", nargs="?", const="all", default=None, choices=("all", "cfg", "nocfg"), help="--engine fused で、DiT の行列積を FP8 にする（engine/fp8.py）")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    set_num_steps(args.steps)
    rt = load_runtime()
    if args.no_watermark:
        from engine.bucket import disable_watermark
        disable_watermark(rt)
    if args.half_decode:
        from engine.fused import half_decode
        half_decode(rt)
    if args.engine == "fused":
        from engine.fused import install_fused
        sampler = install_fused(rt, precision=args.precision, num_steps=args.steps, **({"fp8": args.fp8} if args.fp8 else {}))
    else:
        sampler = install_bucket(rt, precision=args.precision, num_steps=args.steps)
    with torch.inference_mode():
        ref_latent, ref_mask = rt._load_reference_latent(req=make_request("あ。", 1.0, 0), batch_size=1, messages=[])
        info = sampler.prewarm(sampler.voice_for(ref_latent, ref_mask))
    info.update(prewarm_codec(rt))
    decoder = None
    if args.bucket_decode:
        from engine.decode import install_bucket_decoder
        decoder = install_bucket_decoder(rt)
        info.update(decoder.prewarm())
        info["device_used_after_prewarm_gb"] = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
    # 範囲のすべての組（潜在 13〜750 フレーム × 文 1〜256 トークン）が、記録ずみのグラフに当たることを数え上げて確かめる
    from engine.bucket import pick_bucket
    n_spk = sampler.voice_for(ref_latent, ref_mask).tokens
    pairs = 0
    for seq_len in range(13, 751):
        for n_text in range(1, 257):
            lb, tb = sampler.bucket_for(seq_len, n_text)
            assert (lb, tb, n_spk) in sampler.shapes, (seq_len, n_text)
            assert pick_bucket(sampler.text_encode_buckets, n_text) in sampler.text_graphs, n_text
            pairs += 1
    info["covered_pairs"] = pairs
    print("covered", pairs, "pairs", flush=True)
    seen = {}
    orig = sampler.sample

    def spy(**kw):
        seen.update(seq_len=int(kw["sequence_length"]), n_text=int(kw["text_mask"].sum()),
                    text_width=int(kw["text_input_ids"].shape[1]))
        out = orig(**kw)
        seen["finite"] = bool(torch.isfinite(out).all())
        return out

    import irodori_tts.inference_runtime as IR
    from engine.fast import runtime_sampler_name
    setattr(IR, runtime_sampler_name(rt.model), spy)
    rows = []
    for label, text, seconds in cases():
        text = normalize_text(text)
        row = {"case": label, "chars": len(text), "seconds": seconds}
        for k in ("first", "second"):
            seen.clear()
            audio, r = timed_synth(rt, make_request(text, seconds, 1))
            row[f"{k}_total_sec"] = r["total_sec"]
            row[f"{k}_sample_rf_sec"] = r["stages"].get("sample_rf", r["stages"].get("sample_meanflow"))
            row[f"{k}_decode_sec"] = r["stages"]["decode_latent"]
            row[f"{k}_watermark_sec"] = r["stages"].get("silentcipher_watermark")
        lb, tb = sampler.bucket_for(seen["seq_len"], seen["n_text"])
        row.update(latent=seen["seq_len"], latent_bucket=lb, text_tokens=seen["n_text"], text_bucket=tb,
                   latent_finite=seen["finite"], audio_sec=r["audio_sec"], audio_finite=bool(np.isfinite(audio).all()),
                   audio_peak=float(np.abs(audio).max()) if len(audio) else 0.0,
                   late_captures=sampler.late_captures, fallbacks=sampler.fallbacks)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    info.update(max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                memory_reserved_gb=torch.cuda.memory_reserved() / 1e9,
                device_used_end_gb=(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9)
    out = {"precision": args.precision, "prewarm": info, "rows": rows,
           "late_captures": sampler.late_captures, "fallbacks": sampler.fallbacks}
    info["device_used_nvidia_smi_note"] = "device_used_end_gb は mem_get_info の値（このプロセス以外も含む GPU 全体の使用量）"
    tag = ("" if args.engine == "bucket" else f"{args.engine}-") + args.precision
    tag += ("-nowm" if args.no_watermark else "") + ("-hdec" if args.half_decode else "") + ("-bdec" if args.bucket_decode else "")
    tag += (f"-fp8{args.fp8 if args.fp8 != 'all' else ''}" if args.fp8 else "") + steps_tag(args.steps) + args.suffix
    p = WORK / "runs/checks" / f"edges-{tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", p, "late_captures", sampler.late_captures, "fallbacks", sampler.fallbacks, flush=True)
    assert sampler.late_captures == 0 and sampler.fallbacks == 0
    assert all(r["latent_finite"] and r["audio_finite"] for r in rows)
    if decoder is not None:
        print("decode cold", decoder.cold, "misses", decoder.misses, flush=True)
        assert decoder.cold == 0 and decoder.misses == 0


if __name__ == "__main__":
    main()
