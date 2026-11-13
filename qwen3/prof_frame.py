"""qwen3/fast.py の 1 フレーム（CUDA Graph の再生）のカーネルの内訳を torch.profiler で見る。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh \
    qwen3/prof_frame.py --model 0.6b [--compile]

1 フレームの壁時計の時間と、カーネルの実行時間の合計（GPU が働いている時間）と、カーネルの数を出す。
カーネルの合計が壁時計より十分に短ければ、カーネルとカーネルのあいだの隙間（起動の手間）が大きい。
カーネルの名前ごとの合計時間の上位も出す（行列とベクトルの積、注意、要素ごとの演算のどれが重いか）。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import LANGUAGE, SPEAKER, load  # noqa: E402
from fast import Qwen3Fast, Sampling  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--engine", choices=["fast", "kernels"], default="fast")
    ap.add_argument("--wd", default="bf16")
    ap.add_argument("--sampler", choices=["fused", "torch"], default="fused")
    ap.add_argument("--l2-keep-mb", type=float, default=None)
    ap.add_argument("--cp-pair", choices=["auto", "on", "off"], default="auto",
                    help="--engine kernels で code predictor の位置 0 と 1 をまとめて通すか（auto は on）")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    tts = load(args.model, "bfloat16", "sdpa")
    probe = Qwen3Fast(tts)
    kw = probe.prompt("今日もゆるっと雑談していきますね。", SPEAKER, LANGUAGE)
    del probe
    if args.engine == "kernels":
        from kernels import Qwen3Kernels

        eng = Qwen3Kernels(tts, Sampling.from_kwargs(kw), wd=args.wd, sampler=args.sampler, l2_keep_mb=args.l2_keep_mb,
                           cp_pair={"auto": None, "on": True, "off": False}[args.cp_pair])
    else:
        eng = Qwen3Fast(tts, Sampling.from_kwargs(kw), compile=args.compile)
    print("capture", round(eng.capture(), 2), flush=True)
    # 1 文を作ったあとの状態から、終わりのトークンに関係なく決まった回数だけ再生する（キャッシュの位置は進み続ける）
    torch.manual_seed(1)
    eng.generate_codes(kw)
    for _ in range(10):
        eng.graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.frames):
        eng.graph.replay()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.frames
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.frames):
            eng.graph.replay()
        torch.cuda.synchronize()
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot[e.name] += e.device_time_total
            cnt[e.name] += 1
    n = sum(cnt.values()) / args.frames
    busy = sum(tot.values()) / args.frames / 1000
    print(f"model {args.model} engine {args.engine} wd {args.wd} compile {args.compile}: wall {wall * 1000:.2f} ms/frame, kernels {busy:.2f} ms/frame, "
          f"{n:.0f} kernels/frame")
    # 最初の入力（HF のモジュール）の時間
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            eng.talker.model(inputs_embeds=kw["inputs_embeds"], past_key_values=eng.tcache,
                             cache_position=torch.arange(kw["inputs_embeds"].shape[1], device=eng.device), use_cache=True)
    torch.cuda.synchronize()
    print(f"prefill（HF、{kw['inputs_embeds'].shape[1]} トークン）: {(time.perf_counter() - t0) / 5 * 1000:.2f} ms")
    for name, t in sorted(tot.items(), key=lambda x: -x[1])[: args.top]:
        print(f"{t / args.frames / 1000:8.3f} ms  {cnt[name] / args.frames:6.1f}x  {name[:110]}")


if __name__ == "__main__":
    main()
