"""1 行の入力（M=1）の行列とベクトルの積（GEMV）が、重みをどれだけの速さで読めるかを測る。Qwen3-TTS のエンジンの下調べ。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/gemv_bench.py

形は、code predictor（5 層、幅 1024）と言語モデル（0.6B: 28 層・幅 1024、1.7B: 28 層・幅 2048）の 1 層の行列
（q・k・v をまとめたもの、o、gate・up をまとめたもの、down）。重みの型は次の 3 つ。
  bf16       : 今の重みのまま
  i8row      : int8 と行ごとの尺度（fp32）
  i8g32      : int8 と、行の中の 32 個ごとの尺度（fp16。speech.cpp の Q8_0 と同じ分け方）
比べるもの
  torch      : torch.nn.functional.linear（cuBLAS）。今のエンジンが使っているもの
  gemv       : 自作のカーネル（1 つの warp が 1 行、16 バイトずつ読む）
code predictor は 1 フレームで 15 回続けて通るので、5 層を 15 回通す時間も測る（重みが L2 に残るかを見る）。
すべて CUDA Graph に記録して再生し、起動の手間を除いた時間を測る。
"""

from __future__ import annotations

import time

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F

_SRC = r'''
static __device__ __forceinline__ float bf(unsigned int u16){ return __uint_as_float(u16 << 16); }
static __device__ __forceinline__ float h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
static __device__ __forceinline__ float warp_sum(float v){
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
static __device__ __forceinline__ uint4 ldnc(const uint4* p){
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// y[n] = sum_k W[n, k] x[k]。W は bf16 の [N, K]（K は 256 の倍数）。1 つの warp が 1 行。
extern "C" __global__ void gemv_bf16(const unsigned short* __restrict__ W, const float* __restrict__ x,
                                     float* __restrict__ y, int N, int K){
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= N) return;
  const uint4* w = (const uint4*)(W + (size_t)row * K);
  int nc = K >> 3;
  float acc = 0.f;
  #pragma unroll 4
  for (int c = lane; c < nc; c += 32){
    uint4 v = ldnc(w + c);
    const float4* xp = (const float4*)(xs + c * 8);
    float4 a = xp[0], b = xp[1];
    acc += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w
         + bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

static __device__ __forceinline__ float dot4i8(unsigned int u, const float* xp){
  return (float)(signed char)(u & 0xffu) * xp[0] + (float)(signed char)((u >> 8) & 0xffu) * xp[1]
       + (float)(signed char)((u >> 16) & 0xffu) * xp[2] + (float)(signed char)(u >> 24) * xp[3];
}

// int8 と行ごとの尺度。
extern "C" __global__ void gemv_i8row(const signed char* __restrict__ W, const float* __restrict__ scale,
                                      const float* __restrict__ x, float* __restrict__ y, int N, int K){
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= N) return;
  const uint4* w = (const uint4*)(W + (size_t)row * K);
  int nc = K >> 4;
  float acc = 0.f;
  #pragma unroll 4
  for (int c = lane; c < nc; c += 32){
    uint4 v = ldnc(w + c);
    const float* xp = xs + c * 16;
    acc += dot4i8(v.x, xp) + dot4i8(v.y, xp + 4) + dot4i8(v.z, xp + 8) + dot4i8(v.w, xp + 12);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc * scale[row];
}

// int8 と、行の中の 32 個ごとの尺度（fp16、[N, K/32]）。16 バイト（16 個）ずつ読むので、2 回に 1 回尺度が変わる。
extern "C" __global__ void gemv_i8g32(const signed char* __restrict__ W, const unsigned short* __restrict__ scale,
                                      const float* __restrict__ x, float* __restrict__ y, int N, int K){
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= N) return;
  const uint4* w = (const uint4*)(W + (size_t)row * K);
  const unsigned short* sc = scale + (size_t)row * (K >> 5);
  int nc = K >> 4;
  float acc = 0.f;
  #pragma unroll 4
  for (int c = lane; c < nc; c += 32){
    uint4 v = ldnc(w + c);
    const float* xp = xs + c * 16;
    float s = h2f(sc[c >> 1]);
    acc += s * (dot4i8(v.x, xp) + dot4i8(v.y, xp + 4) + dot4i8(v.z, xp + 8) + dot4i8(v.w, xp + 12));
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}
'''

GPU_BW_NOTE = "RTX 6000 Ada の仕様上の帯域は 960 GB/s"


def shapes(hidden: int, inter: int, q: int = 2048, kv: int = 1024) -> list[tuple[str, int, int]]:
    return [("qkv", q + 2 * kv, hidden), ("o", hidden, q), ("gate_up", 2 * inter, hidden), ("down", hidden, inter)]


def main() -> None:
    dev = torch.device("cuda")
    mod = cp.RawModule(code=_SRC, options=("-std=c++14", "--use_fast_math"), name_expressions=None)
    k = {n: mod.get_function(n) for n in ("gemv_bf16", "gemv_i8row", "gemv_i8g32")}
    ptr = lambda t: np.uintp(t.data_ptr())  # noqa: E731
    WARPS = 8

    def make_layer(hidden, inter):
        out = []
        for name, n, kk in shapes(hidden, inter):
            w = (torch.randn(n, kk, device=dev) * 0.02).to(torch.bfloat16)
            wf = w.float()
            srow = wf.abs().amax(1).clamp_min(1e-8) / 127
            qrow = torch.round(wf / srow[:, None]).clamp(-127, 127).to(torch.int8)
            g = wf.view(n, kk // 32, 32)
            sg = (g.abs().amax(2).clamp_min(1e-8) / 127).half()
            qg = torch.round(g / sg.float()[..., None]).clamp(-127, 127).to(torch.int8).view(n, kk)
            out.append({"name": name, "N": n, "K": kk, "w": w, "qrow": qrow, "srow": srow.contiguous(),
                        "qg": qg.contiguous(), "sg": sg.contiguous().view(torch.int16)})
        return out

    xb = torch.randn(1, 8192, device=dev).to(torch.bfloat16)

    def run(layers, kind, x, y):
        s = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with s:
            for L in layers:
                for m in L:
                    n, kk = m["N"], m["K"]
                    if kind == "torch":
                        F.linear(xb[:, :kk], m["w"])
                        continue
                    grid = ((n + WARPS - 1) // WARPS,)
                    blk = (32 * WARPS,)
                    smem = kk * 4
                    if kind == "bf16":
                        k["gemv_bf16"](grid, blk, (ptr(m["w"]), ptr(x), ptr(y), np.int32(n), np.int32(kk)), shared_mem=smem)
                    elif kind == "i8row":
                        k["gemv_i8row"](grid, blk, (ptr(m["qrow"]), ptr(m["srow"]), ptr(x), ptr(y), np.int32(n), np.int32(kk)),
                                        shared_mem=smem)
                    else:
                        k["gemv_i8g32"](grid, blk, (ptr(m["qg"]), ptr(m["sg"]), ptr(x), ptr(y), np.int32(n), np.int32(kk)),
                                        shared_mem=smem)

    def nbytes(layers, kind):
        b = 0
        for L in layers:
            for m in L:
                n, kk = m["N"], m["K"]
                b += n * kk * (2 if kind in ("bf16", "torch") else 1) + (n * 4 if kind == "i8row" else 0) \
                    + (n * kk // 32 * 2 if kind == "i8g32" else 0)
        return b

    def timed(layers, kind, reps):
        x = torch.randn(8192, device=dev)
        y = torch.zeros(16384, device=dev)
        seq = layers * reps
        run(seq, kind, x, y)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(seq, kind, x, y)
        g.replay()
        torch.cuda.synchronize()
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            g.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        return dt, nbytes(seq, kind) / dt / 1e9

    # 正しさ: 自作の GEMV と torch の結果を比べる
    L = make_layer(1024, 3072)
    x = torch.randn(8192, device=dev)
    y = torch.zeros(16384, device=dev)
    for m in L:
        n, kk = m["N"], m["K"]
        ref = m["w"].float() @ x[:kk]
        for kind in ("bf16", "i8row", "i8g32"):
            run([[m]], kind, x, y)
            torch.cuda.synchronize()
            err = (y[:n] - ref).norm() / ref.norm()
            print(f"check {m['name']:8s} {kind:6s} 相対誤差 {err:.2e}")
    print(GPU_BW_NOTE)
    cpl = [make_layer(1024, 3072) for _ in range(5)]
    t06 = [make_layer(1024, 3072) for _ in range(28)]
    t17 = [make_layer(2048, 6144) for _ in range(28)]
    for title, layers, reps in (("code predictor 5 層 × 1 回", cpl, 1), ("code predictor 5 層 × 15 回", cpl, 15),
                                ("言語モデル 0.6B 28 層", t06, 1), ("言語モデル 1.7B 28 層", t17, 1),
                                ("0.6B: 言語モデル + code predictor × 15", t06 + cpl * 15, 1),
                                ("1.7B: 言語モデル + code predictor × 15", t17 + cpl * 15, 1)):
        for kind in ("torch", "bf16", "i8row", "i8g32"):
            dt, bw = timed(layers, kind, reps)
            print(f"{title:40s} {kind:6s} {dt * 1000:7.3f} ms  {bw:7.1f} GB/s", flush=True)


if __name__ == "__main__":
    main()
