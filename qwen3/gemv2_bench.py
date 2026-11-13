"""qwen3/kernels.py の GEMV（rms_gemv、gemv）を、1 フレームと同じ形の並びで測る。1 版目のカーネルとも比べる。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/gemv2_bench.py [--wd i8g32]

比べるもの
  v1       : 1 版目の GEMV（このファイルの V1。1 つの warp が 1〜2 行、ブロック全体の RMSNorm、__syncthreads 3 回、
             L2 の扱いを指定しない読み込み）
  v2-nohint: 今の GEMV（最初の重みを先に読む、__syncthreads 1 回）で、L2 の扱いを指定しない読み込み（-DNO_L2_HINT）
  v2       : 今の GEMV。L2 の扱いを指定する（policy 0: 普通に残す、1: 先に追い出す）
並びは attention を除いた 1 層の 4 つの GEMV（qkv、o、gate・up、down）。重みは乱数（形は本物と同じ）。
  code predictor だけ : 5 層 × 16 回（1 フレームで code predictor を通る回数）
  1 フレーム           : 言語モデル 28 層 + code predictor 5 層 × 16 回。言語モデルの重みが L2 を通り抜ける
すべて CUDA Graph に記録して再生した時間。L2 の大きさと、L2 の一部を残す用途に取っておける上限も出す。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import _SRC, _Lin  # noqa: E402

# 1 版目の GEMV（kernels.py の 1 版目から写したもの。名前に _v1 を付けた）
V1 = r'''
template <int RPW>
static __device__ __forceinline__ void rows_v1(const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                               int row0, int K, const float* xs, float* acc){
  int lane = threadIdx.x & 31;
  int nc = K / CHUNK;
  #pragma unroll
  for (int r = 0; r < RPW; r++) acc[r] = 0.f;
  #pragma unroll 2
  for (int c = lane; c < nc; c += 32){
    uint4 v[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) v[r] = ldnc((const uint4*)(W + (size_t)(row0 + r) * K) + c);
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] += dotc(wconv(v[r]), xs + c * CHUNK, LDS(S, row0 + r, K, c));
  }
  #pragma unroll
  for (int r = 0; r < RPW; r++) acc[r] = warp_sum(acc[r]);
}
extern "C" __global__ void rms_gemv_v1(const float* __restrict__ h, const float* __restrict__ nw, float eps,
                                       const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                       float* __restrict__ out, int N, int K, int swiglu){
  extern __shared__ float xs[];
  __shared__ float red1[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x){ float v = h[i]; xs[i] = v; ss += v * v; }
  float inv = rsqrtf(block_sum(ss, red1) / K + eps);
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xs[i] * inv * nw[i];
  __syncthreads();
  int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 2;
  if (row0 >= N) return;
  float acc[2];
  rows_v1<2>(W, S, row0, K, xs, acc);
  if ((threadIdx.x & 31) == 0){
    if (swiglu){ float g = acc[0]; out[row0 >> 1] = g / (1.f + expf(-g)) * acc[1]; }
    else { out[row0] = acc[0]; out[row0 + 1] = acc[1]; }
  }
}
extern "C" __global__ void gemv_v1(const float* __restrict__ x, const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                   const float* __restrict__ bias, float* __restrict__ out, int N, int K, int accumulate){
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (row >= N) return;
  float acc[1];
  rows_v1<1>(W, S, row, K, xs, acc);
  if ((threadIdx.x & 31) == 0){
    float v = acc[0];
    if (bias) v += bias[row];
    if (accumulate) v += out[row];
    out[row] = v;
  }
}
'''


def ptr(t):
    return np.uintp(0 if t is None else t.data_ptr())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", default="i8g32", choices=["i8g32", "bf16"])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    opts = ("-std=c++14",) + {"bf16": (), "i8g32": ("-DWD_I8G32",)}[args.wd]
    hint = cp.RawModule(code=_SRC + V1, options=opts)
    nohint = cp.RawModule(code=_SRC, options=opts + ("-DNO_L2_HINT",))
    k = {"v1": (hint.get_function("rms_gemv_v1"), hint.get_function("gemv_v1")),
         "v2": (hint.get_function("rms_gemv"), hint.get_function("gemv")),
         "v2-nohint": (nohint.get_function("rms_gemv"), nohint.get_function("gemv"))}
    torch.manual_seed(0)

    def layer(H, I, q=2048, kv=1024):
        mk = lambda n, kk: _Lin((torch.randn(n, kk, device=dev) * 0.02).to(torch.bfloat16), args.wd)  # noqa: E731
        return {"qkv": mk(q + 2 * kv, H), "o": mk(H, q), "gu": mk(2 * I, H), "down": mk(H, I),
                "nw": (1 + 0.1 * torch.randn(H, device=dev)).contiguous()}

    def rms(ver, h, nw, lin, out, swiglu, pol):
        f = k[ver][0]
        if ver == "v1":
            f((-(-lin.N // 16),), (256,), (ptr(h), ptr(nw), np.float32(1e-6), ptr(lin.w), ptr(lin.s), ptr(out),
                                           np.int32(lin.N), np.int32(lin.K), np.int32(swiglu)), shared_mem=lin.K * 4)
        else:
            f((-(-lin.N // 16),), (256,),
              (ptr(h), ptr(nw), np.float32(1e-6), ptr(lin.w), ptr(lin.s), ptr(out), np.int32(lin.N), np.int32(lin.K),
               np.int32(swiglu), np.int32(1), np.int32(pol)), shared_mem=lin.K * 4)

    def gemv(ver, x, lin, out, acc, pol):
        f = k[ver][1]
        if ver == "v1":
            f((-(-lin.N // 8),), (256,), (ptr(x), ptr(lin.w), ptr(lin.s), ptr(None), ptr(out), np.int32(lin.N),
                                          np.int32(lin.K), np.int32(acc)), shared_mem=lin.K * 4)
        else:
            f((-(-lin.N // 8),), (256,), (ptr(x), ptr(lin.w), ptr(lin.s), ptr(None), ptr(out), np.int32(lin.N),
                                          np.int32(lin.K), np.int32(acc), np.int32(1), np.int32(pol)), shared_mem=lin.K * 4)

    bufs = {n: torch.randn(8192, device=dev) * 0.1 for n in ("h", "qkv", "att", "act")}

    def seq(items, ver):
        b = bufs
        for L, pol in items:
            rms(ver, b["h"], L["nw"], L["qkv"], b["qkv"], 0, pol)
            gemv(ver, b["att"], L["o"], b["h"], 1, pol)
            rms(ver, b["h"], L["nw"], L["gu"], b["act"], 1, pol)
            gemv(ver, b["act"], L["down"], b["h"], 1, pol)

    def timed(items, ver):
        g = torch.cuda.CUDAGraph()
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            seq(items, ver)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                seq(items, ver)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.reps * 1000

    # 正しさ: v1 と v2 の出力を比べる（足す順だけが違う）
    L = layer(1024, 3072)
    x = torch.randn(8192, device=dev)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        for name, sw in (("qkv", 0), ("gu", 1)):
            o = {v: torch.zeros(8192, device=dev) for v in ("v1", "v2")}
            for v in o:
                rms(v, x, L["nw"], L[name], o[v], sw, 0)
            torch.cuda.synchronize()
            print(f"check rms_gemv {name}: v1 と v2 の相対差 {float((o['v1'] - o['v2']).norm() / o['v1'].norm()):.2e}")
        for name in ("o", "down"):
            o = {v: torch.zeros(8192, device=dev) for v in ("v1", "v2")}
            for v in o:
                gemv(v, x, L[name], o[v], 0, 0)
            torch.cuda.synchronize()
            print(f"check gemv {name}: v1 と v2 の相対差 {float((o['v1'] - o['v2']).norm() / o['v1'].norm()):.2e}")

    import cupy.cuda.runtime as rt
    dv = torch.cuda.current_device()
    l2 = rt.deviceGetAttribute(38, dv)  # cudaDevAttrL2CacheSize
    pmax = rt.deviceGetAttribute(108, dv)  # cudaDevAttrMaxPersistingL2CacheSize
    cpl = [layer(1024, 3072) for _ in range(5)]
    t06 = [layer(1024, 3072) for _ in range(28)]
    t17 = [layer(2048, 6144) for _ in range(28)]
    cpb = sum(L[n].bytes() for L in cpl for n in ("qkv", "o", "gu", "down"))
    print(f"{torch.cuda.get_device_name()}: L2 {l2 / 2**20:.0f} MiB（残す用途に取っておける上限 {pmax / 2**20:.0f} MiB）、"
          f"重み {args.wd}、code predictor の 5 層の重み {cpb / 2**20:.1f} MiB", flush=True)
    rows = [("code predictor だけ", "v1", [(L, 0) for L in cpl] * 16),
            ("code predictor だけ", "v2-nohint", [(L, 0) for L in cpl] * 16),
            ("code predictor だけ", "v2", [(L, 0) for L in cpl] * 16)]
    for title, tl in (("1 フレーム 0.6B", t06), ("1 フレーム 1.7B", t17)):
        rows += [(title, "v1", [(L, 0) for L in tl] + [(L, 0) for L in cpl] * 16),
                 (title, "v2-nohint", [(L, 0) for L in tl] + [(L, 0) for L in cpl] * 16),
                 (title + "（言語モデルも普通に残す）", "v2", [(L, 0) for L in tl] + [(L, 0) for L in cpl] * 16),
                 (title + "（言語モデルは先に追い出す）", "v2", [(L, 1) for L in tl] + [(L, 0) for L in cpl] * 16),
                 (title + " の言語モデルだけ（先に追い出す）", "v2", [(L, 1) for L in tl])]
    for title, ver, items in rows:
        nbytes = sum(L[n].bytes() for L, _ in items for n in ("qkv", "o", "gu", "down"))
        t = timed(items, ver)
        print(f"{title:36s} {ver:10s} {t:7.3f} ms  {nbytes / t / 1e6:7.1f} GB/s（重みを読んだ量 / 時間）", flush=True)


if __name__ == "__main__":
    main()
