"""バケットとステップごとのグラフの設計を決めるための計測（作った形だけを使う。48 文は使わない）。

- 記録の時間: 1 つのグラフに入れるステップ数（steps_per_graph）、attention の呼び方（cuDNN が先か）、温めの回数ごと
- 再生の時間: 形ごとのグラフ（FastSampler）と比べる。潜在の長さと文のバケットを変えたときの時間

  scripts/run_in_image.sh engine/bucket_probe.py --precision bf16 --part capture
  scripts/run_in_image.sh engine/bucket_probe.py --precision bf16 --part curve
出力: runs/experiments/bucket-probe-<precision>-<part>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
sys.path.insert(0, "/work")
from common import SPEAKER_SCALE, TEXT_SCALE, WORK, load_runtime  # noqa: E402
from engine.bucket import BucketSampler  # noqa: E402
from engine.fast import FastSampler  # noqa: E402


def kwargs(model, seq_len: int, n_text: int, ref_latent, seed: int = 1) -> dict:
    dev = model.device
    g = torch.Generator(device="cpu").manual_seed(n_text)
    ids = torch.zeros((1, 256), dtype=torch.long)
    ids[:, :n_text] = torch.randint(5, 1000, (1, n_text), generator=g)
    mask = torch.zeros((1, 256), dtype=torch.bool)
    mask[:, :n_text] = True
    return dict(model=model, text_input_ids=ids.to(dev), text_mask=mask.to(dev), ref_latent=ref_latent,
                ref_mask=torch.ones(ref_latent.shape[:2], dtype=torch.bool, device=dev), sequence_length=seq_len,
                caption_input_ids=None, caption_mask=None, speaker_uncond_mode="mask", num_steps=40,
                cfg_scale_text=TEXT_SCALE, cfg_scale_caption=3.0, cfg_scale_speaker=SPEAKER_SCALE,
                cfg_guidance_mode="independent", cfg_min_t=0.5, cfg_max_t=1.0, seed=seed)


def timeit(fn, reps: int) -> dict:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"med_ms": float(np.median(ts) * 1e3), "min_ms": float(min(ts) * 1e3), "max_ms": float(max(ts) * 1e3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--part", default="capture", choices=("capture", "curve", "grid"))
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    rt = load_runtime()
    model = rt.model
    g = torch.Generator(device="cpu").manual_seed(0)
    ref_latent = torch.randn(1, 197, 32, generator=g).to(model.device)
    out: dict = {"precision": args.precision, "part": args.part}
    with torch.inference_mode():
        if args.part == "capture":
            exact = FastSampler(model, precision=args.precision)
            kw = kwargs(model, 89, 12, ref_latent)
            out["exact_L89_n12"] = timeit(lambda: exact.sample(**kw), args.reps)
            out["exact_capture_sec"] = exact.capture_sec
            print("exact", out["exact_L89_n12"], exact.capture_sec, flush=True)
            del exact
            rows = []
            for k, cudnn, warm in ((1, False, "first"), (1, False, "all"), (1, True, "first"), (1, True, "all"),
                                   (2, False, "first"), (5, False, "first"), (10, False, "first"), (20, False, "first")):
                # 詰め物なしの形（89, 12）と、既定の区切りに近い形（96, 24）
                s = BucketSampler(model, precision=args.precision, steps_per_graph=k, cudnn_attention=cudnn, warm=warm,
                                  latent_buckets=[89, 96, 104, 112, 120, 128], text_buckets=[12, 24])
                voice = s.voice_for(ref_latent, kw["ref_mask"])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                s.prewarm(voice)
                cap = time.perf_counter() - t0
                r = {"k": k, "cudnn": cudnn, "warm": warm, "shapes": len(s.shapes), "prewarm_sec": cap,
                     "per_shape_sec": float(np.median(s.capture_sec[len(s.text_graphs):])),
                     "nopad_L89_n12": timeit(lambda: s.sample(**kw), args.reps)}
                kw2 = kwargs(model, 96, 24, ref_latent)
                r["full_L96_n24"] = timeit(lambda: s.sample(**kw2), args.reps)
                s.latent_buckets, s.text_buckets = [96, 104, 112, 120, 128], [24]
                r["pad_L89_n12_in_96_24"] = timeit(lambda: s.sample(**kw), args.reps)
                assert s.fallbacks == 0 and s.late_captures == 0
                rows.append(r)
                print(json.dumps(r), flush=True)
                del s
                torch.cuda.empty_cache()
            out["rows"] = rows
        elif args.part == "grid":
            # 細かい区切りで、時間が長さに対してどう変わるかと、形の数に対するメモリと記録の時間を見る。
            lens = list(range(16, 257, 4))
            texts = [8, 16, 24, 32, 40, 48, 64]
            used = lambda: (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9  # noqa: E731
            s = BucketSampler(model, precision=args.precision, latent_buckets=lens, text_buckets=texts)
            voice = s.voice_for(ref_latent, torch.ones((1, 197), dtype=torch.bool, device=model.device))
            torch.cuda.synchronize()
            out["before"] = {"allocated_gb": torch.cuda.memory_allocated() / 1e9, "device_used_gb": used()}
            out["prewarm"] = s.prewarm(voice)
            out["after"] = {"allocated_gb": torch.cuda.memory_allocated() / 1e9, "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                            "device_used_gb": used()}
            print(out, flush=True)
            rows = []
            for tb in (24, 8, 48):
                for L in lens:
                    kw = kwargs(model, L, min(tb, 12), ref_latent)
                    s.text_buckets = [tb]
                    r = {"L": L, "tb": tb, **timeit(lambda: s.sample(**kw), 8)}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
            s.text_buckets = texts
            for L in (56, 88, 120):
                for tb in texts:
                    kw = kwargs(model, L, tb, ref_latent)
                    r = {"L": L, "tb": tb, "exact_text": True, **timeit(lambda: s.sample(**kw), 8)}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
            assert s.fallbacks == 0 and s.late_captures == 0
            out["rows"] = rows
            out["max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        else:
            lens = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512, 750]
            texts = [24, 48, 96, 160, 256]
            s = BucketSampler(model, precision=args.precision, latent_buckets=lens, text_buckets=texts)
            voice = s.voice_for(ref_latent, torch.ones((1, 197), dtype=torch.bool, device=model.device))
            torch.cuda.reset_peak_memory_stats()
            out["prewarm"] = s.prewarm(voice)
            out["capture_sec"] = s.capture_sec
            out["mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            out["mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            print(out, flush=True)
            rows = []
            for L in lens:
                kw = kwargs(model, L, 12, ref_latent)
                r = {"L": L, "tb": 24, **timeit(lambda: s.sample(**kw), args.reps if L < 300 else 10)}
                rows.append(r)
                print(json.dumps(r), flush=True)
            for tb in texts[1:]:
                kw = kwargs(model, 96, tb, ref_latent)
                r = {"L": 96, "tb": tb, **timeit(lambda: s.sample(**kw), args.reps)}
                rows.append(r)
                print(json.dumps(r), flush=True)
            assert s.fallbacks == 0 and s.late_captures == 0
            for n in (4, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
                kw = kwargs(model, 96, n, ref_latent)
                s.sample(**kw)
                tg = s.text_graphs[min(b for b in s.text_encode_buckets if b >= n)]
                r = {"text_encode_n": n, **timeit(lambda: tg["graph"].replay(), args.reps)}
                rows.append(r)
                print(json.dumps(r), flush=True)
            out["rows"] = rows
            out["max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    d = WORK / "runs" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"bucket-probe-{args.precision}-{args.part}.json").write_text(json.dumps(out, indent=1))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
