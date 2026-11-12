"""DiT の 12 層分の行列積だけを、本番と同じ形で測る（cuBLAS、fp32）。

設計メモの見込みは、3 つ分を通すときに FP32 の性能の 5〜6 割、1 つ分のときにメモリ帯域の大半が出ると仮定していた。
公式の実装のプロファイルでは、行列積は 3 つ分で 1 回 5.8 ミリ秒（約 25 TFLOPS）だった。この差を、行列積だけで確かめる。

- 重みは 12 層それぞれ別に持つ（計 268M 個、fp32 で約 1.07 GB）。L2（96 MiB）に収まらないので、毎回メモリから読む。
- 並べ方は 2 通り。separate は公式と同じ 8 個（wq, wk, wv, gate, wo, w1, w3, w2）。
  fused は、入力が同じ行列積をまとめた 4 個（wq/wk/wv/gate を 1280→5120、wo、w1/w3 を 1280→7360、w2）。
- 12 層分を 1 つの CUDA Graph に記録して再生し、CUDA の event で測る。CPU の起動の時間は入らない。
- 行の数: 90（1 つ分、潜在の長さ 90）、270（3 つ分）、1024、4096。精度（fp32、TF32、BF16 の重み）と、cuBLAS と cuBLASLt も比べる。

出力: runs/profile/gemm_bench.json と標準出力の表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/ref")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DIM, HID, LAYERS = 1280, 3680, 12
ROWS = (90, 270, 1024, 4096)
REPLAYS = 30
PARAMS = LAYERS * (5 * DIM * DIM + 3 * DIM * HID)


def make_weights(device):
    g = torch.Generator(device=device).manual_seed(0)
    layers = []
    for _ in range(LAYERS):
        qkvg = torch.randn(4 * DIM, DIM, device=device, generator=g) * 0.02
        wo = torch.randn(DIM, DIM, device=device, generator=g) * 0.02
        w13 = torch.randn(2 * HID, DIM, device=device, generator=g) * 0.02
        w2 = torch.randn(DIM, HID, device=device, generator=g) * 0.02
        layers.append((qkvg, wo, w13, w2))
    return layers


def forward(layers, x, h, layout):
    outs = []
    for qkvg, wo, w13, w2 in layers:
        if layout == "fused":
            outs += [F.linear(x, qkvg), F.linear(x, wo), F.linear(x, w13), F.linear(h, w2)]
        else:  # 公式と同じ 8 個。重みは同じ領域の行の切り出し（連続）を使う。
            outs += [F.linear(x, qkvg[i * DIM:(i + 1) * DIM]) for i in range(4)]
            outs += [F.linear(x, wo), F.linear(x, w13[:HID]), F.linear(x, w13[HID:]), F.linear(h, w2)]
    return outs


def bench(layers, rows, layout, device, dtype=torch.float32):
    x = torch.randn(rows, DIM, device=device, dtype=dtype)
    h = torch.randn(rows, HID, device=device, dtype=dtype)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            forward(layers, x, h, layout)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(layers, x, h, layout)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / REPLAYS
    del graph
    return ms


def main() -> None:
    device = torch.device("cuda")
    fp32_layers = make_weights(device)
    bf16_layers = [tuple(w.to(torch.bfloat16) for w in ws) for ws in fp32_layers]
    results = []
    print(f"{'blas':9s} {'prec':5s} {'layout':9s} {'rows':>5s} {'ms/12層':>9s} {'TFLOPS':>7s} {'重み GB/s':>9s}")
    configs = [(b, p) for b in ("cublas", "cublaslt") for p in ("fp32", "tf32")] + [("cublas", "bf16")]
    for blas, prec in configs:
        torch.backends.cuda.preferred_blas_library(blas)
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        layers, dtype, nbytes = (bf16_layers, torch.bfloat16, 2) if prec == "bf16" else (fp32_layers, torch.float32, 4)
        for layout in ("separate", "fused"):
            for rows in ROWS:
                ms = bench(layers, rows, layout, device, dtype)
                tflops = 2 * rows * PARAMS / (ms * 1e-3) / 1e12
                gbps = PARAMS * nbytes / (ms * 1e-3) / 1e9
                r = {"blas": blas, "prec": prec, "layout": layout, "rows": rows, "ms": ms,
                     "tflops": tflops, "weight_gbps": gbps}
                results.append(r)
                print(f"{blas:9s} {prec:5s} {layout:9s} {rows:5d} {ms:9.3f} {tflops:7.1f} {gbps:9.0f}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path("/work/runs/profile/gemm_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": PARAMS, "device": torch.cuda.get_device_name(0),
                               "torch": torch.__version__, "results": results}, indent=1))


if __name__ == "__main__":
    main()
