"""イメージの中で動くエンジンの速さを、48 文 × シード 1, 2 で測る（作者の前の計測の bench_tts.py と同じ範囲）。

測る範囲: 合成 → 24kHz への変換 → 前後の無音の切り落とし（音声合成サーバーの X-Synthesis-Seconds と同じ）。
各条件の前に 2 文で温める。--engine graphs は、48 文の長さのグラフをすべて記録してから測る。

  scripts/run_in_image.sh ref/bench_engine.py --engine official
  scripts/run_in_image.sh ref/bench_engine.py --engine graphs
  scripts/run_in_image.sh ref/bench_engine.py --engine graphs-trim   # パディングを attention から外す
  scripts/run_in_image.sh ref/bench_engine.py --engine fast --precision tf32 --out-dir bench/stage1   # engine/fast.py

  scripts/run_in_image.sh ref/bench_engine.py --engine bucket --precision bf16 --out-dir bench/stage1b   # engine/bucket.py。起動の準備だけで測る
  scripts/run_in_image.sh ref/bench_engine.py --engine bucket --precision bf16 --out-dir bench/stage1b --warm48   # 48 文を一度通してから測る
  scripts/run_in_image.sh ref/bench_engine.py --engine bucket --precision bf16 --out-dir bench/nowm --no-watermark   # 電子透かしを外す
  IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh ref/bench_engine.py --engine fused --precision bf16 --out-dir bench/fused --no-watermark   # engine/fused.py（cupy の入ったイメージが要る）
  （同じく）... --engine fused --no-watermark --bucket-decode                # 復号を決まった長さで動かす（engine/decode.py）
  scripts/bench_compare.sh bench/bdec 2 "cc:--engine bucket" "cc:--engine fused --bucket-decode"   # 条件を交互に、順に測る

--engine bucket と fused は、起動の時間・グラフの数・メモリー・途中の記録の回数を <engine>.info.json に書く。途中の記録か公式の関数への
切り替えが 1 回でもあれば、最後に止まる。

ステップ数の既定は 20（ref/common.py の DEFAULT_STEPS。2026-11-11 に利用者が決めた）。名前に -s20 が付く。
-s の付かない前の記録は 40 ステップなので、比べるときは --steps 40 を付けて測る。
MF 版は IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF で選ぶ（既定 4 ステップ。名前に -mf が付く）。
--engine graphs と graphs-trim は RF 版だけ。

出力: runs/<out-dir>/<engine>.jsonl（1 行 1 合成）。集計は tools/bench_summary.py。
--out-dir を変えると、前の計測（runs/bench/）を上書きせずに測り直せる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/work/ref")
from common import (DEFAULT_STEPS, WORK, load_runtime, load_sentences, make_request, normalize_text, set_num_steps, steps_tag,  # noqa: E402
                    timed_synth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("official", "graphs", "graphs-trim", "fast", "bucket", "fused"), default="official")
    ap.add_argument("--warm48", action="store_true",
                    help="--engine bucket で、測る前に 48 文を一度通す（すぐの速さ）。付けなければ、起動の準備だけで測る（冷えた状態）")
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"),
                    help="--engine fast の精度。tf32 は DiT の行列積だけ（engine/fast.py）")
    ap.add_argument("--out-dir", default="bench", help="runs/ の下の出力先")
    ap.add_argument("--suffix", default="", help="出力のファイル名の後ろに付ける（同じ条件をもう一度測るとき）")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--tf32", action="store_true", help="行列積の TF32 を有効にする（公式の既定は無効）")
    ap.add_argument("--no-watermark", action="store_true",
                    help="電子透かし（SilentCipher）を外す（engine/bucket.py の disable_watermark）。ファイル名に -nowm が付く")
    ap.add_argument("--half-decode", action="store_true",
                    help="復号を float16 の autocast で動かす（engine/fused.py の half_decode。出力が変わる）。ファイル名に -hdec が付く")
    ap.add_argument("--bucket-decode", action="store_true",
                    help="復号を決まった長さで動かし、起動のときにすべて通す（engine/decode.py）。ファイル名に -bdec が付く")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="1 文ごとに GPU を休ませる秒数。本番のように間が空くときの速さを見る（電力の上限でクロックが下がるため）")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"サンプリングのステップ数（既定 {DEFAULT_STEPS}。ref/common.py の DEFAULT_STEPS）。CFG は同じ時刻の範囲（0.5〜1.0）にかかる。"
                         "40 以外はファイル名に -s<数> が付く（-s のない記録は 40 ステップ）")
    ap.add_argument("--fp8", nargs="?", const="all", default=None, choices=("all", "cfg", "nocfg"),
                    help="--engine fused で、DiT の行列積を FP8（e4m3）にする（engine/fp8.py。出力が変わる）。ファイル名に -fp8 が付く")
    args = ap.parse_args()
    set_num_steps(args.steps)
    rt = load_runtime()
    if args.no_watermark:
        sys.path.insert(0, "/work")
        from engine.bucket import disable_watermark

        disable_watermark(rt)
    if args.half_decode:
        sys.path.insert(0, "/work")
        from engine.fused import half_decode

        half_decode(rt)
    if args.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
    bucketlike = args.engine in ("bucket", "fused")  # 起動の準備だけで測るエンジン
    name = args.engine + (f"-{args.precision}" if args.engine in ("fast", "bucket", "fused") else "")
    name += ("" if not bucketlike else "-warm" if args.warm48 else "-cold") + ("-tf32" if args.tf32 else "")
    name += (f"-gap{args.gap:g}" if args.gap else "") + ("-nowm" if args.no_watermark else "")
    name += ("-hdec" if args.half_decode else "") + ("-bdec" if args.bucket_decode else "")
    name += (f"-fp8{args.fp8 if args.fp8 != 'all' else ''}" if args.fp8 else "") + steps_tag(args.steps) + args.suffix
    assert not args.fp8 or args.engine == "fused", "--fp8 は --engine fused だけ"
    decoder = None
    sentences = load_sentences()
    extra = {}
    sampler = None
    if bucketlike:
        # 起動の準備: 本番の範囲を覆うバケットをすべて記録する。48 文の形は使わない。
        import torch
        sys.path.insert(0, "/work")
        from engine.bucket import install_bucket, prewarm_codec

        if args.engine == "fused":
            from engine.fused import install_fused as install_bucket

        used = lambda: (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9  # noqa: E731
        t0 = time.perf_counter()
        ekw = {"num_steps": args.steps, **({"fp8": args.fp8} if args.fp8 else {})}
        sampler = install_bucket(rt, precision=args.precision, **ekw)
        with torch.inference_mode():
            ref_latent, ref_mask = rt._load_reference_latent(req=make_request("あ。", 1.0, 0), batch_size=1, messages=[])
            before = used()
            info = sampler.prewarm(sampler.voice_for(ref_latent, ref_mask))
        info.update(prewarm_codec(rt))
        if args.bucket_decode:
            from engine.decode import install_bucket_decoder

            decoder = install_bucket_decoder(rt)
            info.update(decoder.prewarm())
        extra = {**info, "startup_sec": time.perf_counter() - t0, "device_used_before_prewarm_gb": before,
                 "device_used_after_prewarm_gb": used(), "latent_buckets": len(sampler.latent_buckets),
                 "text_buckets": len(sampler.text_buckets), "text_encode_buckets": len(sampler.text_encode_buckets),
                 "kernel_compile_sec": getattr(sampler, "kernel_compile_sec", None), "num_steps": sampler.num_steps,
                 "cfg_steps": sampler.n_cfg}
        if args.warm48:
            for s in sentences:
                timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], 0))
        print("prewarmed", extra, flush=True)
    elif args.engine == "fast":
        assert not args.tf32, "--engine fast では --precision tf32 を使う（全体の --tf32 は使わない）"
        sys.path.insert(0, "/work")
        from engine.fast import install

        sampler = install(rt, precision=args.precision, num_steps=args.steps)
        t0 = time.perf_counter()
        for s in sentences:
            timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], 0))
        assert sampler.fallbacks == 0
        extra = {"graphs": len(sampler.graphs), "capture_total_sec": sum(sampler.capture_sec),
                 "warm_sec": time.perf_counter() - t0}
        print("captured", extra, flush=True)
    elif args.engine.startswith("graphs"):
        assert str(rt.model_cfg.flow_parameterization) != "meanflow", "--engine graphs は RF 版だけ"
        import irodori_tts.inference_runtime as IR
        import irodori_tts.model as M
        from graphs_baseline import GraphSampler, _timestep_embedding

        M.get_timestep_embedding = _timestep_embedding
        rt.model._rope_freqs(2048, rt.model.device)
        sampler = GraphSampler(rt.model, trim_padding=args.engine == "graphs-trim")
        IR.sample_euler_rf_cfg = sampler.sample
        t0 = time.perf_counter()
        for s in sentences:
            timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], 0))
        extra = {"graphs": len(sampler.graphs), "capture_total_sec": sum(sampler.capture_sec),
                 "warm_sec": time.perf_counter() - t0}
        print("captured", extra, flush=True)
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        timed_synth(rt, make_request(text, 2.0, 0))
    out = WORK / "runs" / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for seed in (int(x) for x in args.seeds.split(",")):
            for i, s in enumerate(sentences):
                text = normalize_text(s["text"])
                if args.gap:
                    time.sleep(args.gap)
                _, r = timed_synth(rt, make_request(text, s["seconds"], seed))
                r.update({"engine": name, "i": i, "seed": seed, "chars": len(text), "seconds": s["seconds"],
                          "watermark": rt.watermarker.ready, **extra})
                if bucketlike:
                    r.update(late_captures=sampler.late_captures, fallbacks=sampler.fallbacks)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if bucketlike:
        import torch
        end = {"late_captures": sampler.late_captures, "fallbacks": sampler.fallbacks,
               "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
               "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
               "device_used_end_gb": (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9, **extra}
        if decoder is not None:
            end.update(decode_cold=decoder.cold, decode_misses=decoder.misses)
        (out / f"{name}.info.json").write_text(json.dumps(end, indent=1))
        print("end", end, flush=True)
        assert sampler.late_captures == 0 and sampler.fallbacks == 0
        assert decoder is None or (decoder.cold == 0 and decoder.misses == 0)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
