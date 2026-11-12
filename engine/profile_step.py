"""エンジンの DiT の 1 回分の GPU の計算時間を、カーネルの種類ごとに出す（torch.profiler）。
あわせて、CUDA Graph の再生だけの時間（文のエンコード + KV + 40 ステップ）を、続けて回して測る。

  scripts/run_in_image.sh engine/profile_step.py --precision bf16
出力: runs/profile/fast-<precision>-kernels.txt、標準出力
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/work/ref")
sys.path.insert(0, "/work")
from common import WORK, load_runtime  # noqa: E402
from engine.fast import FastSampler, matmul_tf32  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"))
    ap.add_argument("--latent", type=int, default=89)
    ap.add_argument("--text", type=int, default=12)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    rt = load_runtime()
    model = rt.model
    dev = model.device
    sampler = FastSampler(model, precision=args.precision)
    tag = args.precision
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.inference_mode():
        ref_latent = torch.randn(1, 197, 32, generator=g).to(dev)
        voice = sampler.voice_for(ref_latent, torch.ones(1, 197, dtype=torch.bool, device=dev))
        ids = torch.randint(5, 1000, (1, args.text), generator=g).to(dev)
        x0 = torch.randn(1, args.latent, 32, generator=g).to(dev)
        # 再生だけの時間
        t0 = time.perf_counter()
        sampler.run(x0, ids, voice)
        torch.cuda.synchronize()
        print(f"first run (capture): {time.perf_counter() - t0:.1f} s", flush=True)
        x1 = torch.randn(1, args.latent + 7, 32, generator=g).to(dev)
        t0 = time.perf_counter()
        sampler.run(x1, ids, voice)
        torch.cuda.synchronize()
        print(f"second shape (capture): {time.perf_counter() - t0:.1f} s", flush=True)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            sampler.run(x0, ids, voice)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"replay {tag} L={args.latent} n={args.text}: med {np.median(times) * 1e3:.1f} ms "
              f"min {min(times) * 1e3:.1f} max {max(times) * 1e3:.1f}", flush=True)
        if args.no_profile:
            return
        text_state = sampler.encode_text(ids)
        mask3 = sampler.cfg_mask(args.latent, args.text, voice.tokens)
        lines = []
        with matmul_tf32(args.precision == "tf32"):
            ctx = sampler.build_context(text_state, voice.kv)
            for name, i in (("cfg(B=3)", 0), ("nocfg(B=1)", 39)):
                for _ in range(3):
                    sampler.step(x0, i, ctx, mask3)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        sampler.step(x0, i, ctx, mask3)
                    torch.cuda.synchronize()
                evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                agg: dict[str, list[float]] = {}
                for e in evs:
                    a = agg.setdefault(e.name, [0.0, 0])
                    a[0] += e.device_time
                    a[1] += 1
                total = sum(a[0] for a in agg.values())
                lines.append(f"== {tag} {name}: GPU {total / 10 / 1e3:.2f} ms/step, kernels {sum(a[1] for a in agg.values()) // 10}/step")
                for k, a in sorted(agg.items(), key=lambda kv: -kv[1][0])[:22]:
                    lines.append(f"  {a[0] / 10 / 1e3:7.3f} ms  x{a[1] // 10:4d}  {k[:110]}")
    out = WORK / "runs" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"fast-{tag}-kernels.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
