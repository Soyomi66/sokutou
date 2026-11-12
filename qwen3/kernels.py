"""Qwen3-TTS の 1 フレームを、自作の CUDA カーネル（CuPy の RawModule、NVRTC）で動かすエンジン。qwen3/fast.py の Qwen3Fast を元にする。

Qwen3Fast は 1 フレームを CUDA Graph にまとめたが、HF のモジュールのままなので 1 フレームに約 7000 個の小さなカーネルが動き、
その数で 15 ms かかる（qwen3/prof_frame.py）。このエンジンは、Transformer の 1 層を次の 5 個のカーネルにまとめる（入力は 1 トークン）。
  1. rms_gemv(qkv)   : RMSNorm と、q・k・v をまとめた重みの行列とベクトルの積（GEMV）
  2. attn            : q と k の頭ごとの RMSNorm、RoPE、K・V のキャッシュへの書き込み、注意（GQA、1 つの kv の頭に q の頭 2 つ）
  3. gemv(o)         : o の GEMV と、残差への足し込み
  4. rms_gemv(gu)    : RMSNorm と、gate・up を 1 行ずつ交互に並べた重みの GEMV、silu(gate) × up
  5. gemv(down)      : down の GEMV と、残差への足し込み
残差は float32 で持ち、行列の積は float32 で足す（公式は bfloat16 で持つ）。重みの型は 2 つから選ぶ（wd）。
  bf16  : 公式の重みのまま
  i8g32 : int8 と、行の中の 32 個ごとの尺度（fp16）。speech.cpp の Q8_0 と同じ分け方。読む量が約半分になる
言語モデルの最初の入力（プロンプト）は、HF のモジュールのまま StaticCache に入れ、そのキャッシュをこのカーネルが読み書きする
（並びは [1, kv の頭, 長さ, 128] で同じ）。code predictor の最初の 2 トークン（言語モデルの隠れ状態と第 0 のコード）は、
1 トークンずつ 2 回通す（公式は 2 トークンをまとめて通す。計算は同じで、丸めの順だけが変わる）。
GEMV は、最初の重みを入力の準備より先に読み始め、RMSNorm の 2 乗和を warp ごとに求める（__syncthreads は 1 回）。
重みを読むときに L2 の扱いを指定する。1 フレームに 16 回読む code predictor の重み（int8 で約 80 MiB）は L2 に残し、
1 回だけ読む重み（言語モデルなど）は先に追い出す。指定しないと、code predictor の重みが L2 にほとんど残らない。
サンプリングは 2 通り（sampler）。
  fused : qwen3/sampler.py の 1 カーネル（processor、温度、top-k、引く）。自前の乱数（Philox）なので、同じシードでも公式とは違う列になる
  torch : Qwen3Fast と同じ torch の演算（公式と同じ順で乱数を使う）。公式とコードの列を突き合わせるとき（qwen3/check.py）に使う
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fast import Qwen3Fast, Sampling, _sample

_SRC = r'''
#define WARPS 8
#define HD 128
typedef unsigned short bf16_t;

static __device__ __forceinline__ float bf(unsigned int u){ return __uint_as_float(u << 16); }
static __device__ __forceinline__ bf16_t f2bf(float f){  // 偶数丸め（PyTorch の BF16 への変換と同じ）
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16_t)(u >> 16);
}
// キャッシュと埋め込みの表の型。ACT_F32 なら float32（float32 のモデルで公式と突き合わせるとき）
#ifdef ACT_F32
typedef float ct_t;
static __device__ __forceinline__ float LDC(ct_t v){ return v; }
static __device__ __forceinline__ ct_t STC(float f){ return f; }
#else
typedef bf16_t ct_t;
static __device__ __forceinline__ float LDC(ct_t v){ return bf(v); }
static __device__ __forceinline__ ct_t STC(float f){ return f2bf(f); }
#endif
static __device__ __forceinline__ float h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
static __device__ __forceinline__ float warp_sum(float v){
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
static __device__ __forceinline__ float warp_max(float v){
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
static __device__ __forceinline__ uint4 ldnc(const uint4* p){
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
// ブロックの全スレッドの和。全スレッドが同じ値を受け取る（足す順は決まっている）。red は 32 個の共有メモリー。
static __device__ __forceinline__ float block_sum(float v, float* red){
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
  __syncthreads();
  return t;
}

#if defined(WD_F32)
#define CHUNK 4
typedef float wt_t;
static __device__ __forceinline__ float dotc(uint4 v, const float* xp, const unsigned short* S, int row, int K, int c){
  float4 a = ((const float4*)xp)[0];
  return __uint_as_float(v.x) * a.x + __uint_as_float(v.y) * a.y + __uint_as_float(v.z) * a.z + __uint_as_float(v.w) * a.w;
}
#elif defined(WD_I8G32)
#define CHUNK 16
typedef signed char wt_t;
static __device__ __forceinline__ float dot4i8(unsigned int u, const float* xp){
  return (float)(signed char)(u & 0xffu) * xp[0] + (float)(signed char)((u >> 8) & 0xffu) * xp[1]
       + (float)(signed char)((u >> 16) & 0xffu) * xp[2] + (float)(signed char)(u >> 24) * xp[3];
}
static __device__ __forceinline__ float dotc(uint4 v, const float* xp, const unsigned short* S, int row, int K, int c){
  float s = h2f(S[(size_t)row * (K >> 5) + (c >> 1)]);
  return s * (dot4i8(v.x, xp) + dot4i8(v.y, xp + 4) + dot4i8(v.z, xp + 8) + dot4i8(v.w, xp + 12));
}
#else
#define CHUNK 8
typedef bf16_t wt_t;
static __device__ __forceinline__ float dotc(uint4 v, const float* xp, const unsigned short* S, int row, int K, int c){
  float4 a = ((const float4*)xp)[0], b = ((const float4*)xp)[1];
  return bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w
       + bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
}
#endif

// 重みの読み込み（L1 に置かない。L2 の扱いは pol で選ぶ）
static __device__ __forceinline__ uint4 ldp(const uint4* p, unsigned long long pol){
#ifdef NO_L2_HINT  // 比べるとき（qwen3/gemv2_bench.py）: L2 の扱いを指定しない読み込み
  return ldnc(p);
#else
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
  return r;
#endif
}
// policy 0: L2 に普通に残す（evict_normal）。1: L2 から先に追い出す（evict_first）。
// 何も指定しない ld.global.nc では、1 フレームに 16 回読む code predictor の重みが L2 にほとんど残らなかった（qwen3/gemv2_bench.py）
static __device__ __forceinline__ unsigned long long mkpol(int policy){
  unsigned long long pol;
  if (policy == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
#define U 4
// 行列とベクトルの積の本体。1 ブロック = 8 warp。warp w は行の組 grp = w / split と、組の中の番号 sub = w % split を受け持つ。
// 組は RPW 行で、1 行の K / CHUNK 個のかたまりを、組の split 個の warp で 32 × split ずつ分けて読む。
// 最初の U 回分の重みを、入力の準備（共有メモリーへの書き込みと RMSNorm の 2 乗和）より先に読み始める。
// nw があれば、共有メモリーに x · nw を置き、最後に 1 / rms を掛ける（2 乗和は warp ごとに x 全体から求める）。
// 結果は組の sub 0 の lane 0 が受け取る（row_out が -1 ならそのスレッドは書かない）。
template <int RPW>
static __device__ __forceinline__ void gemv_core(const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                                 const float* __restrict__ src, const float* __restrict__ nw, float eps,
                                                 int N, int K, int split, int policy, float* acc_out, int* row_out){
  extern __shared__ float xs[];
  __shared__ float red[WARPS][RPW];
  const unsigned long long pol = mkpol(policy);
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gpb = WARPS / split, grp = w / split, sub = w - grp * split;
  const int row0 = (blockIdx.x * gpb + grp) * RPW;
  const bool act = row0 < N;
  const int nc = K / CHUNK, stride = 32 * split, c0 = sub * 32 + lane;
  const uint4* wr[RPW];
  #pragma unroll
  for (int r = 0; r < RPW; r++) wr[r] = (const uint4*)(W + (size_t)(act ? row0 + r : 0) * K);
  uint4 v[U][RPW];
  #pragma unroll
  for (int u = 0; u < U; u++){
    int c = c0 + u * stride;
    #pragma unroll
    for (int r = 0; r < RPW; r++) if (act && c < nc) v[u][r] = ldp(wr[r] + c, pol);
  }
  float inv = 1.f;
  if (nw){
    float ss = 0.f;
    for (int i = lane * 4; i < K; i += 128){ float4 a = *(const float4*)(src + i); ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w; }
    inv = rsqrtf(warp_sum(ss) / K + eps);
    for (int i = threadIdx.x * 4; i < K; i += blockDim.x * 4){
      float4 a = *(const float4*)(src + i), b = *(const float4*)(nw + i);
      *(float4*)(xs + i) = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }
  } else {
    for (int i = threadIdx.x * 4; i < K; i += blockDim.x * 4) *(float4*)(xs + i) = *(const float4*)(src + i);
  }
  __syncthreads();
  float acc[RPW];
  #pragma unroll
  for (int r = 0; r < RPW; r++) acc[r] = 0.f;
  if (act){
    for (int base = c0; base < nc; base += U * stride){
      #pragma unroll
      for (int u = 0; u < U; u++){
        int c = base + u * stride;
        #pragma unroll
        for (int r = 0; r < RPW; r++) if (c < nc) acc[r] += dotc(v[u][r], xs + c * CHUNK, S, row0 + r, K, c);
      }
      #pragma unroll
      for (int u = 0; u < U; u++){
        int c = base + (U + u) * stride;
        #pragma unroll
        for (int r = 0; r < RPW; r++) if (c < nc) v[u][r] = ldp(wr[r] + c, pol);
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < RPW; r++) acc[r] = warp_sum(acc[r]) * inv;
  if (split > 1){
    if (lane == 0){
      #pragma unroll
      for (int r = 0; r < RPW; r++) red[w][r] = acc[r];
    }
    __syncthreads();
    if (sub == 0){
      #pragma unroll
      for (int r = 0; r < RPW; r++){
        float t = 0.f;
        for (int q = 0; q < split; q++) t += red[grp * split + q][r];
        acc[r] = t;
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < RPW; r++) acc_out[r] = acc[r];
  *row_out = (act && sub == 0 && lane == 0) ? row0 : -1;
}

// out = W · RMSNorm(h)。swiglu が 1 なら、行 2n（gate）と 2n+1（up）から out[n] = silu(gate) × up。1 つの組が 2 行。
extern "C" __global__ void rms_gemv(const float* __restrict__ h, const float* __restrict__ nw, float eps,
                                    const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                    float* __restrict__ out, int N, int K, int swiglu, int split, int policy){
  float acc[2]; int row0;
  gemv_core<2>(W, S, h, nw, eps, N, K, split, policy, acc, &row0);
  if (row0 < 0) return;
  if (swiglu){ float g = acc[0]; out[row0 >> 1] = g / (1.f + expf(-g)) * acc[1]; }
  else { out[row0] = acc[0]; out[row0 + 1] = acc[1]; }
}

// out = W · x (+ bias)（accumulate が 1 なら out に足す）。1 つの組が 1 行。
extern "C" __global__ void gemv(const float* __restrict__ x, const wt_t* __restrict__ W, const unsigned short* __restrict__ S,
                                const float* __restrict__ bias, float* __restrict__ out, int N, int K, int accumulate,
                                int split, int policy){
  float acc[1]; int row;
  gemv_core<1>(W, S, x, nullptr, 0.f, N, K, split, policy, acc, &row);
  if (row < 0) return;
  float v = acc[0];
  if (bias) v += bias[row];
  if (accumulate) v += out[row];
  out[row] = v;
}

// out = RMSNorm(h)（1 ブロック）。言語モデルの最後の隠れ状態（code predictor に渡す）を作る。
extern "C" __global__ void rmsnorm(const float* __restrict__ h, const float* __restrict__ nw, float eps, float* __restrict__ out, int K){
  __shared__ float red[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x){ float v = h[i]; ss += v * v; }
  float inv = rsqrtf(block_sum(ss, red) / K + eps);
  for (int i = threadIdx.x; i < K; i += blockDim.x) out[i] = h[i] * inv * nw[i];
}

// out = table[idx[0]]（bf16 の埋め込みの 1 行を float32 で）。
extern "C" __global__ void embed(const ct_t* __restrict__ table, const long long* __restrict__ idx, float* __restrict__ out, int dim){
  long long r = idx[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x)
    out[i] = LDC(table[(size_t)r * dim + i]);
}

// 言語モデルの入力: out = Σ_g tabs[g][codes[g]] + pad（g は 0〜G-1。tabs は埋め込みの表の先頭のアドレスの配列）。
extern "C" __global__ void talker_in(const unsigned long long* __restrict__ tabs, const long long* __restrict__ codes,
                                     const ct_t* __restrict__ pad, float* __restrict__ out, int dim, int G){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x){
    float s = 0.f;
    for (int g = 0; g < G; g++) s += LDC(((const ct_t*)tabs[g])[(size_t)codes[g] * dim + i]);
    out[i] = s + LDC(pad[i]);
  }
}

// 注意（入力は 1 トークン）。1 ブロックが 1 つの kv の頭と、それを使う q の頭 2 つを受け持つ（256 スレッド = 2 × HD）。
// qkv は [q の頭 × HD | k の頭 × HD | v の頭 × HD]。位置は pos_ptr[0] + pos_add（pos_ptr が 0 なら pos_add）。
// rope は [位置][cos 64 個 | sin 64 個]。Kc・Vc はこの層の [kv の頭][max_len][HD]（bf16）。out は [q の頭 × HD]。
extern "C" __global__ void attn(const float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                                float eps, const float* __restrict__ rope, ct_t* __restrict__ Kc, ct_t* __restrict__ Vc,
                                const long long* __restrict__ pos_ptr, int pos_add, int max_len, int nq, int nkv,
                                float* __restrict__ out){
  extern __shared__ float sc[];  // [2][max_len]
  __shared__ float qs[2][HD];
  __shared__ float ks[HD];
  __shared__ float red[8];
  __shared__ float stat[4];
  const int kvh = blockIdx.x, t = threadIdx.x, part = t >> 7, d = t & (HD - 1), w = t >> 5, l = t & 31;
  const int pos = (pos_ptr ? (int)pos_ptr[0] : 0) + pos_add;
  const float c = rope[(size_t)pos * HD + (d & 63)], s = rope[(size_t)pos * HD + 64 + (d & 63)];
  // k（part 0）と v（part 1）
  float kv = qkv[nq * HD + part * nkv * HD + kvh * HD + d];
  float sq = warp_sum(kv * kv);
  if (l == 0) red[w] = sq;
  __syncthreads();
  float tot = red[part * 4] + red[part * 4 + 1] + red[part * 4 + 2] + red[part * 4 + 3];
  __syncthreads();
  if (part == 0) ks[d] = kv * rsqrtf(tot / HD + eps) * kn[d];
  __syncthreads();
  if (part == 0){
    float rot = d < 64 ? -ks[d + 64] : ks[d - 64];
    Kc[((size_t)kvh * max_len + pos) * HD + d] = STC(ks[d] * c + rot * s);
  } else {
    Vc[((size_t)kvh * max_len + pos) * HD + d] = STC(kv);
  }
  // q の頭 kvh*2 + part
  const int qh = kvh * 2 + part;
  float qv = qkv[qh * HD + d];
  sq = warp_sum(qv * qv);
  if (l == 0) red[w] = sq;
  __syncthreads();
  tot = red[part * 4] + red[part * 4 + 1] + red[part * 4 + 2] + red[part * 4 + 3];
  qs[part][d] = qv * rsqrtf(tot / HD + eps) * qn[d];
  __syncthreads();
  float rq = d < 64 ? -qs[part][d + 64] : qs[part][d - 64];
  float qr = (qs[part][d] * c + rq * s) * 0.08838834764831845f;  // 1 / sqrt(HD)
  __syncthreads();
  qs[part][d] = qr;
  __syncthreads();  // ここまでで、この位置の K・V の書き込みもブロックの全スレッドから見える
  // 点数: warp w が位置 j = w, w+8, ... を受け持ち、lane が 4 次元ずつ
  const int T = pos + 1;
  const ct_t* Kh = Kc + (size_t)kvh * max_len * HD;
  const float4 qa = ((const float4*)qs[0])[l], qb = ((const float4*)qs[1])[l];
  for (int j = w; j < T; j += 8){
    const ct_t* kp = Kh + (size_t)j * HD + 4 * l;
    float k0 = LDC(kp[0]), k1 = LDC(kp[1]), k2 = LDC(kp[2]), k3 = LDC(kp[3]);
    float s0 = warp_sum(qa.x * k0 + qa.y * k1 + qa.z * k2 + qa.w * k3);
    float s1 = warp_sum(qb.x * k0 + qb.y * k1 + qb.z * k2 + qb.w * k3);
    if (l == 0){ sc[j] = s0; sc[max_len + j] = s1; }
  }
  __syncthreads();
  // softmax（part ごとに 128 スレッド = 4 warp）
  float* sp = sc + part * max_len;
  float m = -__int_as_float(0x7f800000);
  for (int j = d; j < T; j += HD) m = fmaxf(m, sp[j]);
  m = warp_max(m);
  if (l == 0) red[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(red[part * 4], red[part * 4 + 1]), fmaxf(red[part * 4 + 2], red[part * 4 + 3]));
  __syncthreads();
  float sum = 0.f;
  for (int j = d; j < T; j += HD){ float e = expf(sp[j] - m); sp[j] = e; sum += e; }
  sum = warp_sum(sum);
  if (l == 0) red[w] = sum;
  __syncthreads();
  sum = red[part * 4] + red[part * 4 + 1] + red[part * 4 + 2] + red[part * 4 + 3];
  // 出力: スレッド (part, d) が Σ_j p_j V[j][d]
  const ct_t* Vh = Vc + (size_t)kvh * max_len * HD;
  float acc = 0.f;
  for (int j = 0; j < T; j++) acc += sp[j] * LDC(Vh[(size_t)j * HD + d]);
  out[qh * HD + d] = acc / sum;
}
'''


def _ptr(t: torch.Tensor | None) -> np.uintp:
    if t is None:
        return np.uintp(0)
    assert t.is_contiguous()
    return np.uintp(t.data_ptr())


class _Lin:
    """行列 1 つ（[N, K]）。wd が i8g32 なら int8 と 32 個ごとの尺度（fp16。尺度は |最大| / 127 を fp16 に丸めたもの）にする。"""

    def __init__(self, w: torch.Tensor, wd: str):
        w = w.detach().contiguous()
        self.N, self.K = w.shape
        self.pol = 0    # L2 の扱い（0: 普通に残す、1: 先に追い出す）。Qwen3Kernels が決める
        self.split = 1  # 1 行を何個の warp で分けて読むか（1、2、4、8）
        assert self.K % 512 == 0, f"K={self.K} は 512 の倍数でなければならない"
        if wd == "bf16":
            self.w, self.s = w.to(torch.bfloat16).contiguous(), None
        elif wd == "f32":
            self.w, self.s = w.float().contiguous(), None
        elif wd == "i8g32":
            g = w.float().view(self.N, self.K // 32, 32)
            sc = (g.abs().amax(2) / 127).clamp_min(1e-12).half()
            q = torch.round(g / sc.float()[..., None]).clamp(-127, 127).to(torch.int8)
            self.w, self.s = q.view(self.N, self.K).contiguous(), sc.view(torch.int16).contiguous()
        else:
            raise ValueError(wd)

    def bytes(self) -> int:
        return self.w.numel() * self.w.element_size() + (0 if self.s is None else self.s.numel() * 2)


class _Stack:
    """Transformer の層の重み（言語モデルか code predictor）を、カーネルの形にしたもの。"""

    def __init__(self, model, wd: str, max_len: int):
        cfg = model.config
        self.H, self.I = cfg.hidden_size, cfg.intermediate_size
        self.nq, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        assert cfg.head_dim == 128 and self.nq == 2 * self.nkv, "head_dim 128、q の頭が kv の頭の 2 倍の形だけを扱う"
        self.eps = float(cfg.rms_norm_eps)
        self.layers = []
        for layer in model.layers:
            a, m = layer.self_attn, layer.mlp
            gu = torch.stack([m.gate_proj.weight, m.up_proj.weight], dim=1).reshape(2 * self.I, self.H)
            self.layers.append({
                "in_norm": layer.input_layernorm.weight.float().contiguous(),
                "qkv": _Lin(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0), wd),
                "qn": a.q_norm.weight.float().contiguous(), "kn": a.k_norm.weight.float().contiguous(),
                "o": _Lin(a.o_proj.weight, wd),
                "post_norm": layer.post_attention_layernorm.weight.float().contiguous(),
                "gu": _Lin(gu, wd), "down": _Lin(m.down_proj.weight, wd),
            })
            assert a.q_proj.bias is None and a.o_proj.bias is None
        self.norm = model.norm.weight.float().contiguous()
        inv = model.rotary_emb.inv_freq.float()
        assert inv.numel() == 64 and float(model.rotary_emb.attention_scaling) == 1.0
        f = torch.arange(max_len, device=inv.device, dtype=torch.float32)[:, None] * inv[None, :]
        self.rope = torch.cat([f.cos(), f.sin()], dim=1).contiguous()  # [位置][cos 64 | sin 64]
        self.max_len = max_len

    def bytes(self) -> int:
        return sum(L[k].bytes() for L in self.layers for k in ("qkv", "o", "gu", "down"))


class Qwen3Kernels(Qwen3Fast):
    """Qwen3Fast の 1 フレーム（_frame）を自作のカーネルで置きかえたもの。プロンプトの作り方、最初の入力、サンプリング、
    CUDA Graph の記録と再生、フレームの回し方は Qwen3Fast のまま。"""

    def __init__(self, tts, sampling: Sampling | None = None, max_cache: int = 1024, wd: str = "bf16", graphs: bool = True,
                 sampler: str = "fused", l2_keep_mb: float | None = None):
        super().__init__(tts, sampling, max_cache=max_cache, compile=False, graphs=graphs)
        if sampler not in ("fused", "torch"):
            raise ValueError(sampler)
        if torch.cuda.get_device_capability(self.device) < (8, 0):
            raise RuntimeError("L2 の扱いを指定する読み込み（createpolicy、ld の L2::cache_hint）は Ampere（sm_80）以降が要る")
        import cupy as cp

        self.cpy = cp
        self.wd = wd
        t0 = time.perf_counter()
        act32 = self.dtype == torch.float32
        if act32 != (wd == "f32"):
            raise ValueError("wd=f32 は float32 のモデル、bf16・i8g32 は bfloat16 のモデルで使う")
        opts = ("-std=c++14",) + {"bf16": (), "i8g32": ("-DWD_I8G32",), "f32": ("-DWD_F32", "-DACT_F32")}[wd]
        mod = cp.RawModule(code=_SRC, options=opts)
        self.k = {n: mod.get_function(n) for n in ("rms_gemv", "gemv", "rmsnorm", "embed", "talker_in", "attn")}
        # L2 の大きさ（cudaDevAttrL2CacheSize）。l2_keep_mb を省いたら、L2 から 16 MiB 引いた分（96 MiB なら 80 MiB）にする
        self.l2_bytes = int(cp.cuda.runtime.deviceGetAttribute(38, self.device.index or 0))
        if l2_keep_mb is None:
            l2_keep_mb = max(0.0, self.l2_bytes / 2**20 - 16)
        self.kernel_compile_sec = time.perf_counter() - t0
        dev = self.device
        with torch.no_grad():
            self.T = _Stack(self.talker.model, wd, max_cache)
            self.C = _Stack(self.cp.model, wd, self.G)
            self.cp_heads = [_Lin(h.weight, wd) for h in self.cp.lm_head]
            self.codec_head = _Lin(self.talker.codec_head.weight, wd)
            proj = self.cp.small_to_mtp_projection
            self.proj = None if isinstance(proj, torch.nn.Identity) else (_Lin(proj.weight, wd), proj.bias.float().contiguous())
            self.temb = self.talker.get_input_embeddings().weight.contiguous()
            self.cemb = [e.weight.contiguous() for e in self.cp.get_input_embeddings()]
            assert self.temb.dtype == self.dtype and all(e.dtype == self.dtype for e in self.cemb)
            self.tabs = torch.tensor([self.temb.data_ptr()] + [e.data_ptr() for e in self.cemb], dtype=torch.int64,
                                     device=dev)
        # L2（RTX 6000 Ada で 96 MiB）の使い方: 1 フレームに 16 回読む code predictor の重みを、前の層から l2_keep_mb まで
        # L2 に残し、1 フレームに 1 回だけ読む重み（言語モデル、code predictor の 15 個の頭、codec_head）と、
        # l2_keep_mb を超えた分の code predictor の重みは、L2 から先に追い出す
        mats = ("qkv", "o", "gu", "down")
        for L in self.T.layers:
            for m in mats:
                L[m].pol = 1
        self.codec_head.pol = 1
        for h in self.cp_heads:
            h.pol = 1
        budget = l2_keep_mb * 2**20
        for L in self.C.layers:
            for m in mats:
                b = L[m].bytes()
                L[m].pol = 0 if b <= budget else 1
                budget -= b if b <= budget else 0
        self.l2_keep_mb = l2_keep_mb
        self.l2_kept_bytes = sum(L[m].bytes() for L in self.C.layers for m in mats if L[m].pol == 0)
        # sampler: fused は qwen3/sampler.py の 1 カーネル（自前の乱数）、torch は Qwen3Fast と同じ torch の演算（公式と同じ乱数）
        self.sampler = sampler
        self.fs = None
        if sampler == "fused":
            from sampler import FusedSampler

            self.fs = FusedSampler(dev)
        Ht, Hc = self.T.H, self.C.H
        f32 = dict(dtype=torch.float32, device=dev)
        self.ph32 = torch.zeros(Ht, **f32)       # 言語モデルの最後の隠れ状態（最後の RMSNorm のあと）
        self.ht = torch.zeros(Ht, **f32)         # 言語モデルの残差
        self.hc = torch.zeros(Hc, **f32)         # code predictor の残差
        self.e = torch.zeros(Ht, **f32)          # 埋め込みの 1 行（射影の前）
        self.qkv_t = torch.zeros((self.T.nq + 2 * self.T.nkv) * 128, **f32)
        self.att_t = torch.zeros(self.T.nq * 128, **f32)
        self.act_t = torch.zeros(self.T.I, **f32)
        self.qkv_c = torch.zeros((self.C.nq + 2 * self.C.nkv) * 128, **f32)
        self.att_c = torch.zeros(self.C.nq * 128, **f32)
        self.act_c = torch.zeros(self.C.I, **f32)
        self.logit_c = torch.zeros((1, self.cp.config.vocab_size), **f32)
        self.logit_t = torch.zeros((1, self.V), **f32)
        bf = dict(dtype=self.dtype, device=dev)  # キャッシュと tts_pad の型はモデルと同じ
        self.ck = torch.zeros((len(self.C.layers), self.C.nkv, self.G, 128), **bf)
        self.cv = torch.zeros_like(self.ck)
        self.pad16 = torch.zeros(Ht, **bf)
        self.tk: list[torch.Tensor] = []
        self.tv: list[torch.Tensor] = []

    # --- 起動 -----------------------------------------------------------------------------------------------
    def _launch(self, name: str, grid: int, block: int, args: tuple, smem: int = 0) -> None:
        conv = tuple(_ptr(a) if (a is None or isinstance(a, torch.Tensor)) else a for a in args)
        self.k[name]((int(grid),), (int(block),), conv, shared_mem=int(smem))

    def _rms_gemv(self, h, nw, eps, lin: _Lin, out, swiglu: bool) -> None:
        self._launch("rms_gemv", -(-lin.N // (8 // lin.split * 2)), 256,
                     (h, nw, np.float32(eps), lin.w, lin.s, out, np.int32(lin.N), np.int32(lin.K), np.int32(int(swiglu)),
                      np.int32(lin.split), np.int32(lin.pol)), smem=lin.K * 4)

    def _gemv(self, x, lin: _Lin, bias, out, accumulate: bool) -> None:
        self._launch("gemv", -(-lin.N // (8 // lin.split)), 256,
                     (x, lin.w, lin.s, bias, out, np.int32(lin.N), np.int32(lin.K), np.int32(int(accumulate)),
                      np.int32(lin.split), np.int32(lin.pol)), smem=lin.K * 4)

    def _layers(self, P: _Stack, h, qkv, att, act, kcache, vcache, pos_ptr, pos_add: int) -> None:
        for i, L in enumerate(P.layers):
            self._rms_gemv(h, L["in_norm"], P.eps, L["qkv"], qkv, False)
            self._launch("attn", P.nkv, 256, (qkv, L["qn"], L["kn"], np.float32(P.eps), P.rope, kcache[i], vcache[i],
                                              pos_ptr, np.int32(pos_add), np.int32(P.max_len), np.int32(P.nq),
                                              np.int32(P.nkv), att), smem=2 * P.max_len * 4)
            self._gemv(att, L["o"], None, h, True)
            self._rms_gemv(h, L["post_norm"], P.eps, L["gu"], act, True)
            self._gemv(act, L["down"], None, h, True)

    def _cp_input(self, src_f32: torch.Tensor | None) -> None:
        """code predictor の入力（hc）を作る。src_f32 は射影の前の値（言語モデルの幅）。None なら self.e を使う。"""
        x = self.e if src_f32 is None else src_f32
        if self.proj is None:
            self.hc.copy_(x)
        else:
            self._gemv(x, self.proj[0], self.proj[1], self.hc, False)

    # --- 1 フレーム ---------------------------------------------------------------------------------------------
    def _after_prefill(self, hidden: torch.Tensor) -> None:
        self.ph32.copy_(hidden[0, -1].float())

    def _frame(self) -> None:
        s = self.s
        st = self.cpy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        codes = self.codes
        cptr = codes.data_ptr()
        with st:
            # code predictor の位置 0: 言語モデルの最後の隠れ状態
            self._cp_input(self.ph32)
            self._layers(self.C, self.hc, self.qkv_c, self.att_c, self.act_c, self.ck, self.cv, None, 0)
            # 位置 1: 第 0 のコードの埋め込み（言語モデルの表）
            codes[:, :1].copy_(self.id0)
            self._launch("embed", 8, 256, (self.temb, self.id0, self.e, np.int32(self.T.H)))
            self._cp_input(None)
            for g in range(1, self.G):  # 位置 g で第 g のコードを作る（lm_head[g - 1]）
                if g > 1:
                    self._launch("embed", 8, 256, (self.cemb[g - 2], np.uintp(cptr + 8 * (g - 1)), self.e, np.int32(self.T.H)))
                    self._cp_input(None)
                self._layers(self.C, self.hc, self.qkv_c, self.att_c, self.act_c, self.ck, self.cv, None, g)
                self._rms_gemv(self.hc, self.C.norm, self.C.eps, self.cp_heads[g - 1], self.logit_c, False)
                if self.fs is not None:
                    self.fs(self.logit_c.view(-1), cptr + 8 * g, s.sub_do_sample, s.sub_temperature, s.sub_top_k)
                else:
                    code = _sample(self.logit_c, s.sub_do_sample, s.sub_temperature, s.sub_top_k)
                    codes[:, g:g + 1].copy_(code)
            # 言語モデル: 16 個のコードの埋め込みの和 + tts_pad
            self._launch("talker_in", 8, 256, (self.tabs, codes, self.pad16, self.ht, np.int32(self.T.H), np.int32(self.G)))
            self._layers(self.T, self.ht, self.qkv_t, self.att_t, self.act_t, self.tk, self.tv, self.pos, 0)
            self._launch("rmsnorm", 1, 256, (self.ht, self.T.norm, np.float32(self.T.eps), self.ph32, np.int32(self.T.H)))
            self._gemv(self.ph32, self.codec_head, None, self.logit_t, False)
            if self.fs is not None:
                # processor と、引いたあとの後始末（next_id0、id0、seen、ngen、pos）もカーネルの中で行う
                self.fs(self.logit_t.view(-1), self.next_id0, s.do_sample, s.temperature, s.top_k,
                        talker=dict(seen=self.seen, rep_pen=s.repetition_penalty, suppress=self.suppress, ngen=self.ngen,
                                    min_new=s.min_new_tokens, eos=self.eos, id0=self.id0, pos=self.pos))
        if self.fs is not None:
            return
        nxt = _sample(self._process(self.logit_t), s.do_sample, s.temperature, s.top_k)
        self.next_id0.copy_(nxt)
        self.seen.scatter_(1, nxt, True)
        self.ngen.add_(1)
        self.pos.add_(1)
        self.id0.copy_(nxt)

    def capture(self) -> float:
        """言語モデルの StaticCache を（短い入力で）用意し、そのテンソルをカーネルに渡せるようにしてから記録する。"""
        with torch.no_grad():
            if not self.tk:
                dummy = torch.zeros((1, 2, self.T.H), dtype=self.dtype, device=self.device)
                self.talker.model(inputs_embeds=dummy, past_key_values=self.tcache,
                                  cache_position=torch.arange(2, device=self.device), use_cache=True)
                for layer in self.tcache.layers:
                    assert layer.keys.shape == (1, self.T.nkv, self.max_cache, 128) and layer.keys.dtype == self.dtype
                    self.tk.append(layer.keys[0])
                    self.tv.append(layer.values[0])
        return super().capture()

    def generate_codes(self, kw: dict, max_frames: int | None = None) -> torch.Tensor:
        self.pad16.copy_(kw["tts_pad_embed"].view(-1))
        if self.fs is not None:
            # 自前の乱数の鍵は torch の CPU の乱数から引く（torch.manual_seed で同じ列になり、続けて呼べば毎回変わる）
            self.fs.set_seed(int(torch.randint(0, 2**62, (1,)).item()))
        return super().generate_codes(kw, max_frames=max_frames)

    @torch.no_grad()
    def teacher_logits(self, kw: dict, codes: torch.Tensor):
        """教師強制: codes（frames, 16）を入れたときの、第 0 のコードの logits（1 + frames 個）と code predictor の logits
        （frames × 15 個）を返す（float32、CPU）。最初の入力は HF のモジュール（_prefill_body と同じ）。qwen3/teacher.py が使う。"""
        L = kw["inputs_embeds"].shape[1]
        self.pad16.copy_(kw["tts_pad_embed"].view(-1))
        out = self.talker.model(inputs_embeds=kw["inputs_embeds"], past_key_values=self.tcache,
                                cache_position=torch.arange(L, device=self.device), use_cache=True)
        hidden = out.last_hidden_state
        l0 = [self.talker.codec_head(hidden)[:, -1, :].float().cpu()]
        self.ph32.copy_(hidden[0, -1].float())
        lcp = []
        cptr = self.codes.data_ptr()
        st = self.cpy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        for f in range(codes.shape[0]):
            self.codes.copy_(codes[f].view(1, -1).to(self.device))
            frame = []
            with st:
                self._cp_input(self.ph32)
                self._layers(self.C, self.hc, self.qkv_c, self.att_c, self.act_c, self.ck, self.cv, None, 0)
                for g in range(1, self.G):
                    if g == 1:
                        self._launch("embed", 8, 256, (self.temb, self.codes, self.e, np.int32(self.T.H)))
                    else:
                        self._launch("embed", 8, 256, (self.cemb[g - 2], np.uintp(cptr + 8 * (g - 1)), self.e,
                                                       np.int32(self.T.H)))
                    self._cp_input(None)
                    self._layers(self.C, self.hc, self.qkv_c, self.att_c, self.act_c, self.ck, self.cv, None, g)
                    self._rms_gemv(self.hc, self.C.norm, self.C.eps, self.cp_heads[g - 1], self.logit_c, False)
                    frame.append(self.logit_c.clone())
                self.pos.fill_(L + f)
                self._launch("talker_in", 8, 256, (self.tabs, self.codes, self.pad16, self.ht, np.int32(self.T.H),
                                                   np.int32(self.G)))
                self._layers(self.T, self.ht, self.qkv_t, self.att_t, self.act_t, self.tk, self.tv, self.pos, 0)
                self._launch("rmsnorm", 1, 256, (self.ht, self.T.norm, np.float32(self.T.eps), self.ph32, np.int32(self.T.H)))
                self._gemv(self.ph32, self.codec_head, None, self.logit_t, False)
            lcp.append(torch.cat(frame, dim=0).cpu())
            l0.append(self.logit_t.clone().cpu())
        return torch.cat(l0, dim=0), torch.stack(lcp, dim=0)

    def weight_bytes(self) -> dict:
        return {"talker": self.T.bytes() + self.codec_head.bytes(), "code_predictor": self.C.bytes(),
                "cp_heads": sum(h.bytes() for h in self.cp_heads)}
