"""Fish Audio Drama 3 の 1 フレームを、自作の CUDA カーネル（CuPy の RawModule、NVRTC）で動かす。

公式の実装（fish-speech）は 1 フレーム（言語モデルの頭 → 音声の塔で 1 フレームのコードを引く → 言語モデル 40 層）を
CUDA Graph にまとめているが、torch の演算のままなので、1 フレームに数千個の小さなカーネルが動く。ここでは、まとまった
処理を 1 つのカーネルにする。重みは公式のモデルのテンソルをそのまま読み（bf16。エキスパートは FP8 と 128 × 128 の尺度）、
キャッシュ（全注意の K・V、Gated DeltaNet の畳み込みと再帰の状態、塔の K・V）も公式のテンソルを読み書きする。
最初の入力（プロンプト）は公式のモジュールで流し、そのあとのフレームをこのカーネルで作る。

カーネル（1 層あたり）
  gemv_seg   : RMSNorm と、同じ入力を使う複数の重み（区切り）の行列とベクトルの積（GEMV）。区切りごとに、出力の書き方を
               選ぶ（0: 書く、1: 足す、2: 2 つの重みの行から silu(a) × b、3: Gated DeltaNet の畳み込み（状態の更新）と silu）
  gdn        : Gated DeltaNet の 1 トークン分（q・k の L2 ノルム、β と減衰、状態の更新、出力、gate つきの RMSNorm）
  attn       : 全注意（q・k の頭ごとの RMSNorm、部分 RoPE、K・V の書き込み、GQA、sigmoid の gate）。塔の注意にも使う
  moe_gu     : 経路の選択（256 個から上位 8 個）と、選んだ 8 個のエキスパートの gate・up（FP8）、silu(gate) × up
  moe_down   : 8 個のエキスパートの down（FP8）と共有のエキスパートの down を重みつきで足し、残差に足す
  tower_prep : 塔の層の入力（コードの埋め込みと隠れ状態のそれぞれの RMSNorm を並べたもの）
サンプリングは 1 回を 1 つのカーネルで行う（sample_main: main token の 2 回の引き（RAS）、sample_code: 9 個のコード）。
乱数は自前の Philox4x32-10 なので、同じシードでも公式とは違う列になる（分布は同じ）。

残差と行列の積の和は float32 で持つ（公式は bf16）。エキスパートの入力も float32 のまま掛ける（公式は入力も FP8 にする）。
重みの型（wd）は bf16（公式の重みのまま）、i8g32（int8 と、行の中の 32 個ごとの尺度（fp16）。speech.cpp の Q8_0 と同じ分け方。
エキスパートのほかの重みを読む量が約半分になる。エキスパートは FP8 のまま）、確かめ用の f32（float32 の層と突き合わせる）。
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch

_SRC = r'''
#define WARPS 8
typedef unsigned short bf16_t;

static __device__ __forceinline__ float bf(unsigned int u){ return __uint_as_float(u << 16); }
static __device__ __forceinline__ bf16_t f2bf(float f){  // 偶数丸め（PyTorch の BF16 への変換と同じ）
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16_t)(u >> 16);
}
// キャッシュ・状態・埋め込みの表の型。ACT_F32 なら float32（float32 の層と突き合わせるとき）
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
static __device__ __forceinline__ float silu(float x){ return x / (1.f + expf(-x)); }
static __device__ __forceinline__ float sigm(float x){ return 1.f / (1.f + expf(-x)); }
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
// ブロックの全スレッドの和・最大。全スレッドが同じ値を受け取る。red は (blockDim / 32) 個以上の共有メモリー
static __device__ __forceinline__ float block_sum(float v, float* red){
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
  __syncthreads();
  return t;
}
static __device__ __forceinline__ float block_max(float v, float* red){
  v = warp_max(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = red[0];
  for (int i = 1; i < (int)(blockDim.x >> 5); i++) t = fmaxf(t, red[i]);
  __syncthreads();
  return t;
}
// policy 0: L2 に普通に残す（evict_normal）。1: L2 から先に追い出す（evict_first）
static __device__ __forceinline__ unsigned long long mkpol(int policy){
  unsigned long long pol;
  if (policy == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
// 重みの読み込み（L1 に置かない。L2 の扱いは pol で選ぶ）
static __device__ __forceinline__ uint4 ldp(const uint4* p, unsigned long long pol){
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
  return r;
}
// FP8（e4m3）4 個 → float 4 個（sm_89 の cvt.rn.f16x2.e4m3x2。fp8 の値は f16 でちょうど表せる）
static __device__ __forceinline__ void fp8x4(unsigned int u, float* f){
  unsigned int lo, hi;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi) : "r"(u));
  f[0] = h2f((unsigned short)(lo & 0xffffu)); f[1] = h2f((unsigned short)(lo >> 16));
  f[2] = h2f((unsigned short)(hi & 0xffffu)); f[3] = h2f((unsigned short)(hi >> 16));
}
static __device__ __forceinline__ float dot16_fp8(uint4 v, const float* x){
  float f[4], s = 0.f;
  fp8x4(v.x, f); s += f[0] * x[0] + f[1] * x[1] + f[2] * x[2] + f[3] * x[3];
  fp8x4(v.y, f); s += f[0] * x[4] + f[1] * x[5] + f[2] * x[6] + f[3] * x[7];
  fp8x4(v.z, f); s += f[0] * x[8] + f[1] * x[9] + f[2] * x[10] + f[3] * x[11];
  fp8x4(v.w, f); s += f[0] * x[12] + f[1] * x[13] + f[2] * x[14] + f[3] * x[15];
  return s;
}
// 1 回の 16 バイトの読み込みに入る重みの数（CHUNK）と、その重みと x の内積
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
  const float s = h2f(S[(size_t)row * (K >> 5) + (c >> 1)]);
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

// ---- gemv_seg ----------------------------------------------------------------------------------------------
// 区切りの表 tab: 1 区切り 10 個の int64 [W, S, W2, S2, out, aux0, aux1, N, mode | (policy << 8), 最初のブロック]
// 1 warp が 2 行を受け持つ（mode 2 では W の行 n と W2 の行 n）。入力 src [K]（nw があれば RMSNorm して nw を掛ける）。
// xn_out があれば、ブロック 0 が正規化した入力を書く（MoE のエキスパートの入力）。
#define U 4
#define SEGW 10
static __device__ __forceinline__ float conv_out(ct_t* st, const ct_t* cw, int r, float a){
  // Gated DeltaNet の畳み込み（幅 4、チャンネルごと）: 状態 [r][3]（古い順）と新しい値 a、そのあと silu。状態を 1 つずらす
  ct_t* s = st + (size_t)r * 3;
  const ct_t* w = cw + (size_t)r * 4;
  const float s0 = LDC(s[0]), s1 = LDC(s[1]), s2 = LDC(s[2]);
  const float y = LDC(w[0]) * s0 + LDC(w[1]) * s1 + LDC(w[2]) * s2 + LDC(w[3]) * a;
  s[0] = s[1]; s[1] = s[2]; s[2] = STC(a);
  return silu(y);
}
extern "C" __global__ void __launch_bounds__(256) gemv_seg(const float* __restrict__ src, const float* __restrict__ nw, float eps,
                                                           float* __restrict__ xn_out, const long long* __restrict__ tab,
                                                           int nseg, int K){
  extern __shared__ float xs[];
  int sg = 0;
  while (sg + 1 < nseg && (long long)blockIdx.x >= tab[(sg + 1) * SEGW + 9]) sg++;
  const long long* T = tab + sg * SEGW;
  const int N = (int)T[7], mode = (int)(T[8] & 255), policy = (int)(T[8] >> 8);
  const unsigned long long pol = mkpol(policy);
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int task = ((int)blockIdx.x - (int)T[9]) * WARPS + w;
  const wt_t* W = (const wt_t*)T[0];
  const unsigned short* S = (const unsigned short*)T[1];
  int ra, rb; bool va, vb;
  const wt_t* Wb; const unsigned short* Sb;
  if (mode == 2){ ra = rb = task; va = vb = task < N; Wb = (const wt_t*)T[2]; Sb = (const unsigned short*)T[3]; }
  else { ra = 2 * task; rb = ra + 1; va = ra < N; vb = rb < N; Wb = W; Sb = S; }
  const uint4* wa = (const uint4*)(W + (size_t)(va ? ra : 0) * K);
  const uint4* wb = (const uint4*)(Wb + (size_t)(vb ? rb : 0) * K);
  const int nc = K / CHUNK;
  uint4 v[U][2];
  #pragma unroll
  for (int u = 0; u < U; u++){
    const int c = lane + u * 32;
    if (c < nc){ if (va) v[u][0] = ldp(wa + c, pol); if (vb) v[u][1] = ldp(wb + c, pol); }
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
  if (xn_out && blockIdx.x == 0) for (int i = threadIdx.x; i < K; i += blockDim.x) xn_out[i] = xs[i] * inv;
  float a0 = 0.f, a1 = 0.f;
  for (int base = lane; base < nc; base += U * 32){
    #pragma unroll
    for (int u = 0; u < U; u++){
      const int c = base + u * 32;
      if (c < nc){
        const float* xp = xs + c * CHUNK;
        if (va) a0 += dotc(v[u][0], xp, S, ra, K, c);
        if (vb) a1 += dotc(v[u][1], xp, Sb, rb, K, c);
      }
    }
    #pragma unroll
    for (int u = 0; u < U; u++){
      const int c = base + (U + u) * 32;
      if (c < nc){ if (va) v[u][0] = ldp(wa + c, pol); if (vb) v[u][1] = ldp(wb + c, pol); }
    }
  }
  a0 = warp_sum(a0) * inv;
  a1 = warp_sum(a1) * inv;
  if (lane != 0) return;
  float* out = (float*)T[4];
  if (mode == 0){ if (va) out[ra] = a0; if (vb) out[rb] = a1; }
  else if (mode == 1){ if (va) out[ra] += a0; if (vb) out[rb] += a1; }
  else if (mode == 2){ if (va) out[ra] = silu(a0) * a1; }
  else {
    ct_t* st = (ct_t*)T[5];
    const ct_t* cw = (const ct_t*)T[6];
    if (va) out[ra] = conv_out(st, cw, ra, a0);
    if (vb) out[rb] = conv_out(st, cw, rb, a1);
  }
}

// ---- Gated DeltaNet（1 トークン）--------------------------------------------------------------------------
// 1 ブロック = v の頭 1 つ（512 スレッド。スレッドは v の列 vc = t % 128 と、k の 32 個ずつの組 kg = t / 128）。
// conv: 畳み込みと silu のあと [q 16 × 128 | k 16 × 128 | v 32 × 128]。v の頭 h は k の頭 h / 2 を使う。
// state: [32][128 (k)][128 (v)]（float32）。out: gate つきの RMSNorm のあと [32 × 128]
extern "C" __global__ void __launch_bounds__(512) gdn(const float* __restrict__ conv, const float* __restrict__ z,
                                                      const float* __restrict__ bv, const float* __restrict__ av,
                                                      const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                      const float* __restrict__ gw, float eps, float* __restrict__ state,
                                                      float* __restrict__ out){
  __shared__ float qs[128], ks[128], red[16], part[4][128];
  const int h = blockIdx.x, kh = h >> 1, t = threadIdx.x, w = t >> 5, lane = t & 31, vc = t & 127, kg = t >> 7;
  float qk = 0.f;
  if (t < 256) qk = conv[(t < 128 ? 0 : 2048) + kh * 128 + (t & 127)];
  float sq = warp_sum(qk * qk);
  if (lane == 0) red[w] = sq;
  __syncthreads();
  if (t < 256){
    const int g = (t >> 7) * 4;
    const float r = rsqrtf(red[g] + red[g + 1] + red[g + 2] + red[g + 3] + 1e-6f);
    if (t < 128) qs[t] = qk * r * 0.08838834764831845f;  // 128 ** -0.5
    else ks[t - 128] = qk * r;
  }
  const float beta = sigm(bv[h]);
  const float xa = av[h] + dt_bias[h];
  const float sp = xa > 20.f ? xa : log1pf(expf(xa));  // softplus（torch と同じく 20 を超えたらそのまま）
  const float g = expf(-expf(A_log[h]) * sp);
  const float vv = conv[4096 + h * 128 + vc];
  float* S = state + (size_t)h * 16384 + vc;
  __syncthreads();
  float s[32], rem = 0.f;
  #pragma unroll
  for (int j = 0; j < 32; j++){ const int k = kg * 32 + j; s[j] = S[k * 128] * g; rem += s[j] * ks[k]; }
  part[kg][vc] = rem;
  __syncthreads();
  const float delta = (vv - (part[0][vc] + part[1][vc] + part[2][vc] + part[3][vc])) * beta;
  __syncthreads();
  float o = 0.f;
  #pragma unroll
  for (int j = 0; j < 32; j++){ const int k = kg * 32 + j; s[j] += ks[k] * delta; o += s[j] * qs[k]; S[k * 128] = s[j]; }
  part[kg][vc] = o;
  __syncthreads();
  o = part[0][vc] + part[1][vc] + part[2][vc] + part[3][vc];
  sq = warp_sum(o * o);
  if (lane == 0) red[w] = sq;
  __syncthreads();
  if (t < 128){
    const float r = rsqrtf((red[0] + red[1] + red[2] + red[3]) / 128.f + eps);
    out[h * 128 + vc] = gw[vc] * (o * r) * silu(z[h * 128 + vc]);
  }
}

// ---- 注意（1 トークン、頭の次元 256、q の頭 16、kv の頭 2）----------------------------------------------------------
// qkv: [q と gate 8192 | k 512 | v 512]。interleave_qg が 1 なら q と gate が頭ごとに交互（言語モデル: 頭 h の q は h*512、gate は
// h*512+256）、0 なら [q 16 × 256 | gate 16 × 256]（塔）。neox が 1 なら前半と後半で回す RoPE（言語モデル）、0 なら隣どうし（塔）。
// どちらも最初の 64 次元だけを回す。rope は [位置][cos 32 | sin 32]。Kc・Vc はこの層の [kv の頭][cap][256]。
// 1 ブロック = q の頭 1 つ（256 スレッド）。いまの位置の k・v は共有メモリーから使い、kv の頭ごとに 1 つのブロックだけが書き込む。
static __device__ __forceinline__ float rope_at(const float* x, int d, const float* rp, int neox){
  if (d >= 64) return x[d];
  if (neox){
    const int i = d & 31;
    const float c = rp[i], s = rp[32 + i];
    return d < 32 ? x[d] * c - x[d + 32] * s : x[d] * c + x[d - 32] * s;
  }
  const int i = d >> 1;
  const float c = rp[i], s = rp[32 + i];
  return (d & 1) ? x[d] * c + x[d - 1] * s : x[d] * c - x[d + 1] * s;
}
extern "C" __global__ void __launch_bounds__(256) attn(const float* __restrict__ qkv, int interleave_qg, int neox,
                                                       const float* __restrict__ qn, const float* __restrict__ kn, float eps,
                                                       const float* __restrict__ rope, ct_t* __restrict__ Kc,
                                                       ct_t* __restrict__ Vc, const long long* __restrict__ pos_ptr,
                                                       int pos_add, int cap, float* __restrict__ out){
  extern __shared__ float sc[];  // [位置]
  __shared__ float qs[256], ks[256], vs[256], tmp[256], red[8];
  const int h = blockIdx.x, kvh = h >> 3, d = threadIdx.x, w = d >> 5, lane = d & 31;
  const int pos = (int)pos_ptr[0] + pos_add;
  const float* rp = rope + (size_t)pos * 64;
  // k と v
  const float kr = qkv[8192 + kvh * 256 + d], vv = qkv[8704 + kvh * 256 + d];
  float ss = block_sum(kr * kr, red);
  tmp[d] = kr * rsqrtf(ss / 256.f + eps) * kn[d];
  __syncthreads();
  const float kk = rope_at(tmp, d, rp, neox);
  __syncthreads();
  ks[d] = kk; vs[d] = vv;
  if ((h & 7) == 0){
    Kc[((size_t)kvh * cap + pos) * 256 + d] = STC(kk);
    Vc[((size_t)kvh * cap + pos) * 256 + d] = STC(vv);
  }
  // q と gate
  const float qr = interleave_qg ? qkv[h * 512 + d] : qkv[h * 256 + d];
  const float gt = interleave_qg ? qkv[h * 512 + 256 + d] : qkv[4096 + h * 256 + d];
  ss = block_sum(qr * qr, red);
  tmp[d] = qr * rsqrtf(ss / 256.f + eps) * qn[d];
  __syncthreads();
  qs[d] = rope_at(tmp, d, rp, neox) * 0.0625f;  // 256 ** -0.5
  __syncthreads();
  // 点数: warp w が位置 j = w, w+8, ... を受け持ち、lane が 8 次元ずつ
  const int T = pos + 1;
  const ct_t* Kh = Kc + (size_t)kvh * cap * 256;
  float q8[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) q8[i] = qs[lane * 8 + i];
  for (int j = w; j < T; j += 8){
    float s = 0.f;
    if (j == pos){
      #pragma unroll
      for (int i = 0; i < 8; i++) s += q8[i] * ks[lane * 8 + i];
    } else {
#ifdef ACT_F32
      const float4* kp = (const float4*)(Kh + (size_t)j * 256 + lane * 8);
      const float4 a = kp[0], b = kp[1];
      s = q8[0] * a.x + q8[1] * a.y + q8[2] * a.z + q8[3] * a.w + q8[4] * b.x + q8[5] * b.y + q8[6] * b.z + q8[7] * b.w;
#else
      const uint4 u = *(const uint4*)(Kh + (size_t)j * 256 + lane * 8);
      s = q8[0] * bf(u.x & 0xffffu) + q8[1] * bf(u.x >> 16) + q8[2] * bf(u.y & 0xffffu) + q8[3] * bf(u.y >> 16)
        + q8[4] * bf(u.z & 0xffffu) + q8[5] * bf(u.z >> 16) + q8[6] * bf(u.w & 0xffffu) + q8[7] * bf(u.w >> 16);
#endif
    }
    s = warp_sum(s);
    if (lane == 0) sc[j] = s;
  }
  __syncthreads();
  float m = -__int_as_float(0x7f800000);
  for (int j = d; j < T; j += 256) m = fmaxf(m, sc[j]);
  m = block_max(m, red);
  float sum = 0.f;
  for (int j = d; j < T; j += 256){ const float e = expf(sc[j] - m); sc[j] = e; sum += e; }
  sum = block_sum(sum, red);
  const ct_t* Vh = Vc + (size_t)kvh * cap * 256;
  float acc = 0.f;
  for (int j = 0; j < pos; j++) acc += sc[j] * LDC(Vh[(size_t)j * 256 + d]);
  acc += sc[pos] * vs[d];
  out[h * 256 + d] = acc / sum * sigm(gt);
}

// ---- MoE ---------------------------------------------------------------------------------------------------
// moe_gu: 1 ブロック = 選んだエキスパート 1 つ（slot）の 8 行の組（gate の行 n と up の行 512+n）。64 ブロックで 1 つのエキスパート。
// router: 経路の logits [256]。各ブロックの warp 0 が上位 8 個を選ぶ（softmax の順は logits の順と同じ）。重みは上位 8 個の
// softmax（公式の norm_topk_prob と同じ。全体の softmax の分母は消える）。xn: 正規化した入力 [2048]。
// w13: [256][1024][2048]（FP8。gate の 512 行、up の 512 行）、s13: [256][8][16]（128 × 128 ごとの尺度）
extern "C" __global__ void __launch_bounds__(256) moe_gu(const float* __restrict__ router, const float* __restrict__ xn,
                                                         const unsigned char* __restrict__ w13, const float* __restrict__ s13,
                                                         float* __restrict__ act, int* __restrict__ ids_out,
                                                         float* __restrict__ wts_out){
  __shared__ float xs[2048];
  __shared__ int s_id[8];
  __shared__ float s_wt[8];
  const int t = threadIdx.x, w = t >> 5, lane = t & 31;
  const int slot = blockIdx.x >> 6, n = ((blockIdx.x & 63) << 3) + w;
  if (w == 0){
    float l[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) l[j] = router[lane * 8 + j];
    for (int k = 0; k < 8; k++){
      float best = -__int_as_float(0x7f800000);
      int bi = 0x7fffffff;
      #pragma unroll
      for (int j = 0; j < 8; j++) if (l[j] > best){ best = l[j]; bi = lane * 8 + j; }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1){
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
      }
      if (lane == 0){ s_id[k] = bi; s_wt[k] = best; }
      if ((bi >> 3) == lane){
        #pragma unroll
        for (int j = 0; j < 8; j++) if (j == (bi & 7)) l[j] = -__int_as_float(0x7f800000);
      }
    }
    __syncwarp();
    if (lane == 0){
      const float m = s_wt[0];
      float e[8], sum = 0.f;
      for (int k = 0; k < 8; k++){ e[k] = expf(s_wt[k] - m); sum += e[k]; }
      for (int k = 0; k < 8; k++) s_wt[k] = e[k] / sum;
      if (blockIdx.x == 0) for (int k = 0; k < 8; k++){ ids_out[k] = s_id[k]; wts_out[k] = s_wt[k]; }
    }
  }
  for (int i = t * 4; i < 2048; i += 1024) *(float4*)(xs + i) = *(const float4*)(xn + i);
  __syncthreads();
  const unsigned long long pol = mkpol(1);
  const int e = s_id[slot];
  const unsigned char* base = w13 + (size_t)e * 1024 * 2048;
  const uint4* ga = (const uint4*)(base + (size_t)n * 2048);
  const uint4* ua = (const uint4*)(base + (size_t)(512 + n) * 2048);
  uint4 vg[4], vu[4];
  #pragma unroll
  for (int u = 0; u < 4; u++){ vg[u] = ldp(ga + lane + 32 * u, pol); vu[u] = ldp(ua + lane + 32 * u, pol); }
  const float* sc = s13 + (size_t)e * 128;
  const int rbg = n >> 7, rbu = (512 + n) >> 7;
  float ag = 0.f, au = 0.f;
  #pragma unroll
  for (int u = 0; u < 4; u++){
    const int c = lane + 32 * u, cb = c >> 3;
    ag += sc[rbg * 16 + cb] * dot16_fp8(vg[u], xs + c * 16);
    au += sc[rbu * 16 + cb] * dot16_fp8(vu[u], xs + c * 16);
  }
  ag = warp_sum(ag); au = warp_sum(au);
  if (lane == 0) act[slot * 512 + n] = silu(ag) * au;
}

// moe_down: 1 warp = 出力の 1 行。x[row] += Σ_k wts[k] · W2[ids[k]][row] · act[k] + sigmoid(sgate) · Wsd[row] · sact
// w2: [256][2048][512]（FP8）、s2: [256][16][4]。Wsd: 共有のエキスパートの down [2048][512]（wd の型）
#define NSC (512 / CHUNK / 32)
extern "C" __global__ void __launch_bounds__(256) moe_down(const float* __restrict__ act, const int* __restrict__ ids,
                                                           const float* __restrict__ wts, const unsigned char* __restrict__ w2,
                                                           const float* __restrict__ s2, const float* __restrict__ sact,
                                                           const float* __restrict__ sgate, const wt_t* __restrict__ wsd,
                                                           const unsigned short* __restrict__ ssd, float* __restrict__ x){
  __shared__ float as[8 * 512 + 512];
  __shared__ int s_id[8];
  __shared__ float s_wt[8];
  const int t = threadIdx.x, w = t >> 5, lane = t & 31, row = blockIdx.x * 8 + w;
  const unsigned long long pol = mkpol(1);
  const uint4* sp = (const uint4*)(wsd + (size_t)row * 512);
  uint4 vs[NSC];
  #pragma unroll
  for (int q = 0; q < NSC; q++) vs[q] = ldp(sp + lane + 32 * q, pol);
  if (t < 8){ s_id[t] = ids[t]; s_wt[t] = wts[t]; }
  for (int i = t * 4; i < 4096; i += 1024) *(float4*)(as + i) = *(const float4*)(act + i);
  for (int i = t * 4; i < 512; i += 1024) *(float4*)(as + 4096 + i) = *(const float4*)(sact + i);
  __syncthreads();
  uint4 ve[8];
  #pragma unroll
  for (int k = 0; k < 8; k++) ve[k] = ldp((const uint4*)(w2 + ((size_t)s_id[k] * 2048 + row) * 512) + lane, pol);
  const int rb = row >> 7, cb = lane >> 3;
  float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; k++) acc += s_wt[k] * s2[((size_t)s_id[k] * 16 + rb) * 4 + cb] * dot16_fp8(ve[k], as + k * 512 + lane * 16);
  float sh = 0.f;
  #pragma unroll
  for (int q = 0; q < NSC; q++){ const int c = lane + 32 * q; sh += dotc(vs[q], as + 4096 + c * CHUNK, ssd, row, 512, c); }
  acc = warp_sum(acc);
  sh = warp_sum(sh);
  if (lane == 0) x[row] += acc + sigm(sgate[0]) * sh;
}

// ---- 埋め込みと正規化 --------------------------------------------------------------------------------------------
// 言語モデルの入力: semantic（main が帯の中）なら (temb[main] + cb[4096 + c1]) / sqrt(2)、ほかは temb[main]。
// あわせて frame_in に frame_out を写す（次のフレームの塔の入力。frame_in はこのカーネルでは読まない）
extern "C" __global__ void embed_backbone(const ct_t* __restrict__ temb, const ct_t* __restrict__ cb,
                                          const long long* __restrict__ frame_out, long long* __restrict__ frame_in,
                                          float* __restrict__ x, int dim, int sem_begin, int sem_end){
  const long long main = frame_out[0];
  const bool sem = main >= sem_begin && main <= sem_end;
  const long long c1 = frame_out[2];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x){
    float v = LDC(temb[(size_t)main * dim + i]);
    if (sem) v = (v + LDC(cb[(size_t)(4096 + c1) * dim + i])) / 1.4142135623730951f;
    x[i] = v;
  }
  if (blockIdx.x == 0 && threadIdx.x < 11) frame_in[threadIdx.x] = frame_out[threadIdx.x];
}

// out = RMSNorm(x) · nw（1 ブロック 1024 スレッド、2048 次元）。pos があれば 1 つ進める（フレームの最後）
extern "C" __global__ void __launch_bounds__(1024) rmsnorm_adv(const float* __restrict__ x, const float* __restrict__ nw, float eps,
                                                               float* __restrict__ out, long long* __restrict__ pos){
  __shared__ float red[32];
  const int t = threadIdx.x;
  const float a = x[t], b = x[t + 1024];
  const float inv = rsqrtf(block_sum(a * a + b * b, red) / 2048.f + eps);
  out[t] = a * inv * nw[t];
  out[t + 1024] = b * inv * nw[t + 1024];
  if (pos && t == 0) pos[0] += 1;
}

// 塔の層の入力 xin [4096] = [enorm(e) | hnorm(h)]（1 ブロック 1024 スレッド、2048 次元）。
// h = fnw ? RMSNorm(src) · fnw : src（fnw は前の層の final_layernorm）。e = semantic ? emb[code] : 0。
// cb があれば（最初の共有の層）、入力のフレーム frame_in が semantic のとき、そのコード c2〜c9 の埋め込みの和を e に足す。
extern "C" __global__ void __launch_bounds__(1024) tower_prep(const float* __restrict__ src, const float* __restrict__ fnw,
                                                              const float* __restrict__ hnw, const ct_t* __restrict__ emb,
                                                              const float* __restrict__ enw, const long long* __restrict__ frame_out,
                                                              int code_idx, const ct_t* __restrict__ cb,
                                                              const long long* __restrict__ frame_in, float eps,
                                                              float* __restrict__ xin, int sem_begin, int sem_end){
  __shared__ float red[32];
  const int t = threadIdx.x;
  float h0 = src[t], h1 = src[t + 1024];
  if (fnw){
    const float inv = rsqrtf(block_sum(h0 * h0 + h1 * h1, red) / 2048.f + eps);
    h0 *= inv * fnw[t]; h1 *= inv * fnw[t + 1024];
  }
  float inv = rsqrtf(block_sum(h0 * h0 + h1 * h1, red) / 2048.f + eps);
  xin[2048 + t] = h0 * inv * hnw[t];
  xin[3072 + t] = h1 * inv * hnw[t + 1024];
  const long long main = frame_out[0];
  const bool sem = main >= sem_begin && main <= sem_end;
  long long code = frame_out[code_idx];
  code = code < 0 ? 0 : (code > 4095 ? 4095 : code);
  float e0 = 0.f, e1 = 0.f;
  if (sem){ e0 = LDC(emb[(size_t)code * 2048 + t]); e1 = LDC(emb[(size_t)code * 2048 + t + 1024]); }
  if (cb){
    const long long mi = frame_in[0];
    if (mi >= sem_begin && mi <= sem_end){
      for (int i = 2; i < 10; i++){
        const size_t r = (size_t)(i * 4096 + frame_in[1 + i]) * 2048;
        e0 += LDC(cb[r + t]); e1 += LDC(cb[r + t + 1024]);
      }
    }
  }
  inv = rsqrtf(block_sum(e0 * e0 + e1 * e1, red) / 2048.f + eps);
  xin[t] = e0 * inv * enw[t];
  xin[t + 1024] = e1 * inv * enw[t + 1024];
}

// ---- サンプリング ------------------------------------------------------------------------------------------------
// 公式の sample_top_k_top_p と同じ分布: 温度で割った値の大きい順に、全体の softmax の累積が top_p を超えたもの、または
// 順位が top_k 以上のものを落とし（1 位は残す）、残りから値に比例して引く。上位 top_k 個（64 個まで）を基数選択で集めて並べ、
// 残ったものの中で argmax(v - log(-log u))（u は自前の Philox の一様乱数）を選ぶ。
#define SB 1024
#define PER 5
#define MAXC 64
static __device__ __forceinline__ unsigned int fkey(float f){
  const unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static __device__ __forceinline__ uint4 philox(uint4 c, uint2 k){
  #pragma unroll
  for (int i = 0; i < 10; i++){
    const unsigned int hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const unsigned int hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u; k.y += 0xBB67AE85u;
  }
  return c;
}
struct DrawSmem {
  unsigned int hist[256];
  unsigned int prefix, need;
  float red[32];
  int cnt, result;
  float cv[MAXC];
  int ci[MAXC], order[MAXC];
};
// l: このスレッドの PER 個の値（温度で割ったもの。V 以上の番号は -inf）。stream: 同じフレームの中の何回目の引きか
static __device__ int draw(const float* l, int V, int top_k, float top_p, uint2 key, unsigned long long cnt,
                           unsigned int stream, DrawSmem& sm){
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  float m = -__int_as_float(0x7f800000);
  #pragma unroll
  for (int j = 0; j < PER; j++) m = fmaxf(m, l[j]);
  m = block_max(m, sm.red);
  float z = 0.f;
  #pragma unroll
  for (int j = 0; j < PER; j++) if (t + j * SB < V) z += expf(l[j] - m);
  z = block_sum(z, sm.red);
  int kk = top_k < 1 ? 1 : top_k;
  if (kk > MAXC) kk = MAXC;
  if (kk > V) kk = V;
  // kk 番目の値の順序キー（8 ビットずつ 4 回）
  if (t == 0){ sm.prefix = 0u; sm.need = (unsigned int)kk; }
  for (int pass = 0; pass < 4; pass++){
    const int shift = 24 - 8 * pass;
    if (t < 256) sm.hist[t] = 0u;
    __syncthreads();
    const unsigned int prefix = sm.prefix, need = sm.need;
    const unsigned int hmask = pass ? (0xffffffffu << (shift + 8)) : 0u;
    #pragma unroll
    for (int j = 0; j < PER; j++){
      if (t + j * SB < V){
        const unsigned int k = fkey(l[j]);
        if ((k & hmask) == prefix) atomicAdd(&sm.hist[(k >> shift) & 255u], 1u);
      }
    }
    __syncthreads();
    if (w == 0){
      unsigned int c[8], s = 0u;
      #pragma unroll
      for (int q = 0; q < 8; q++){ c[q] = sm.hist[255 - (lane * 8 + q)]; s += c[q]; }
      unsigned int inc = s;
      #pragma unroll
      for (int o = 1; o < 32; o <<= 1){ const unsigned int y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
      const unsigned int before = inc - s;
      if (before < need && inc >= need){
        unsigned int acc = before;
        #pragma unroll
        for (int q = 0; q < 8; q++){
          if (acc + c[q] >= need){
            const unsigned int digit = 255u - (unsigned int)(lane * 8 + q);
            sm.prefix = prefix | (digit << shift);
            sm.need = need - acc;
            break;
          }
          acc += c[q];
        }
      }
    }
    __syncthreads();
  }
  const unsigned int thr = sm.prefix;
  if (t == 0) sm.cnt = 0;
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < PER; j++){
    const int i = t + j * SB;
    if (i < V && fkey(l[j]) >= thr){
      const int s = atomicAdd(&sm.cnt, 1);
      if (s < MAXC){ sm.cv[s] = l[j]; sm.ci[s] = i; }
    }
  }
  __syncthreads();
  const int n = sm.cnt < MAXC ? sm.cnt : MAXC;
  if (t < n){
    const float v = sm.cv[t];
    const int id = sm.ci[t];
    int r = 0;
    for (int q = 0; q < n; q++){ const float u = sm.cv[q]; r += (u > v) || (u == v && sm.ci[q] < id); }
    sm.order[r] = t;
  }
  __syncthreads();
  if (w == 0){
    float best = -__int_as_float(0x7f800000), carry = 0.f;
    int bi = 0x7fffffff;
    for (int base = 0; base < n; base += 32){
      const int r = base + lane;
      float p = 0.f, v = 0.f;
      int id = 0x7fffffff;
      if (r < n){ const int q = sm.order[r]; v = sm.cv[q]; id = sm.ci[q]; p = expf(v - m) / z; }
      float inc = p;
      #pragma unroll
      for (int o = 1; o < 32; o <<= 1){ const float y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
      const float cum = carry + inc;
      carry += __shfl_sync(0xffffffffu, inc, 31);
      if (r < n && (r == 0 || (cum <= top_p && r < kk))){
        const uint4 rr = philox(make_uint4((unsigned int)id, (unsigned int)cnt, (unsigned int)(cnt >> 32), stream), key);
        const float u = ((float)(rr.x >> 8) + 0.5f) * (1.0f / 16777216.0f);
        const float g = v - logf(-logf(u));
        if (g > best || (g == best && id < bi)){ best = g; bi = id; }
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1){
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
    }
    if (lane == 0) sm.result = bi == 0x7fffffff ? 0 : bi;
  }
  __syncthreads();
  return sm.result;
}
// params: [温度, top_p, top_k] の float32
static __device__ __forceinline__ float clamp_temp(float t){ return t < 1e-5f ? 1e-5f : t; }

// main token: logits [V = 1 + 4096]（0 番が eos、i 番が semantic_begin + i - 1）。1 回目（要求の設定）と 2 回目（RAS: 温度 1、
// top_k 30、top_p 0.95）を引き、1 回目が semantic で直近 16 個の main token（history）にあれば 2 回目を使う。
// frame_out[0] = main、frame_out[1] = main - semantic_begin（0〜4095 に丸める）。history を 1 つずらして main を足す。
// host_main[step] に main を書く（CPU が EOS を見る）。logits に有限でない値があれば bad に 1 を書く。
extern "C" __global__ void __launch_bounds__(1024) sample_main(const float* __restrict__ logits, int V, const float* __restrict__ params,
                                                               const unsigned long long* __restrict__ seed,
                                                               const unsigned long long* __restrict__ counter,
                                                               long long* __restrict__ history, long long* __restrict__ frame_out,
                                                               volatile long long* __restrict__ host_main,
                                                               const long long* __restrict__ step, int* __restrict__ bad,
                                                               int eos, int sem_begin, int sem_end){
  __shared__ DrawSmem sm;
  const int t = threadIdx.x;
  float raw[PER], l[PER];
  bool fin = true;
  #pragma unroll
  for (int j = 0; j < PER; j++){
    const int i = t + j * SB;
    raw[j] = i < V ? logits[i] : -__int_as_float(0x7f800000);
    if (i < V && !isfinite(raw[j])) fin = false;
  }
  if (!fin) atomicOr(bad, 1);
  const unsigned long long sd = seed[0], cnt = counter[0];
  const uint2 key = make_uint2((unsigned int)sd, (unsigned int)(sd >> 32));
  const float temp = clamp_temp(params[0]);
  #pragma unroll
  for (int j = 0; j < PER; j++) l[j] = raw[j] / temp;
  const int d1 = draw(l, V, (int)params[2], params[1], key, cnt, 0u, sm);
  const int d2 = draw(raw, V, 30, 0.95f, key, cnt, 1u, sm);
  if (t == 0){
    const long long m1 = d1 == 0 ? eos : sem_begin + d1 - 1, m2 = d2 == 0 ? eos : sem_begin + d2 - 1;
    bool rep = false;
    if (m1 >= sem_begin && m1 <= sem_end) for (int i = 0; i < 16; i++) rep |= history[i] == m1;
    const long long mn = rep ? m2 : m1;
    for (int i = 0; i < 15; i++) history[i] = history[i + 1];
    history[15] = mn;
    frame_out[0] = mn;
    long long c0 = mn - sem_begin;
    frame_out[1] = c0 < 0 ? 0 : (c0 > 4095 ? 4095 : c0);
    host_main[step[0]] = mn;
  }
}

// 第 slot - 1 のコード（slot は 2〜10）: logits [4096] から引いて frame_out[slot] に書く。last なら、frame_out をフレームの記録
// frames[step] に写し、step と counter を 1 つ進める
extern "C" __global__ void __launch_bounds__(1024) sample_code(const float* __restrict__ logits, const float* __restrict__ params,
                                                               const unsigned long long* __restrict__ seed,
                                                               unsigned long long* __restrict__ counter,
                                                               long long* __restrict__ frame_out, int slot, int last,
                                                               long long* __restrict__ frames, long long* __restrict__ step,
                                                               int* __restrict__ bad){
  __shared__ DrawSmem sm;
  const int t = threadIdx.x;
  const int V = 4096;
  float l[PER];
  bool fin = true;
  const float temp = clamp_temp(params[0]);
  #pragma unroll
  for (int j = 0; j < PER; j++){
    const int i = t + j * SB;
    const float r = i < V ? logits[i] : -__int_as_float(0x7f800000);
    if (i < V && !isfinite(r)) fin = false;
    l[j] = r / temp;
  }
  if (!fin) atomicOr(bad, 1);
  const unsigned long long sd = seed[0], cnt = counter[0];
  const uint2 key = make_uint2((unsigned int)sd, (unsigned int)(sd >> 32));
  const int c = draw(l, V, (int)params[2], params[1], key, cnt, (unsigned int)slot, sm);
  if (t == 0){
    frame_out[slot] = c;
    if (last){
      const long long s = step[0];
      for (int i = 0; i < 11; i++) frames[s * 11 + i] = frame_out[i];
      step[0] = s + 1;
      counter[0] = cnt + 1ull;
    }
  }
}
'''

SEM_BEGIN, SEM_END, EOS = 248099, 252194, 248046
WIDTH = 11  # フレームの行 [main, c0..c9]


def _ptr(t) -> np.uintp:
    if t is None:
        return np.uintp(0)
    if isinstance(t, int):
        return np.uintp(t)
    assert t.is_contiguous(), "連続していないテンソル"
    return np.uintp(t.data_ptr())


class Lin:
    """行列 1 つ（[N, K]）。bf16 は公式のテンソルをそのまま使い（写さない）、f32 は float32 のテンソル。
    i8g32 は int8 と 32 個ごとの尺度（fp16。尺度は |最大| / 127 を fp16 に丸めたもの。qwen3/kernels.py と同じ）。"""

    def __init__(self, w: torch.Tensor, wd: str, pol: int = 1):
        self.N, self.K = w.shape
        assert self.K % 512 == 0, f"K={self.K} は 512 の倍数でなければならない"
        self.s, self.pol = None, pol
        if wd == "bf16":
            assert w.dtype == torch.bfloat16
            self.w = w if w.is_contiguous() else w.contiguous()
        elif wd == "f32":
            self.w = w.float().contiguous()
        elif wd == "i8g32":
            self.w, self.s = quant_i8g32(w)
        else:
            raise ValueError(wd)

    def bytes(self) -> int:
        return self.w.numel() * self.w.element_size() + (0 if self.s is None else self.s.numel() * 2)


def quant_i8g32(w: torch.Tensor, rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] を int8 [N, K] と尺度 [N, K / 32]（fp16 のビットを int16 で持つ）にする。float32 にするのは rows 行ずつ。"""
    N, K = w.shape
    q = torch.empty(N, K, dtype=torch.int8, device=w.device)
    sc = torch.empty(N, K // 32, dtype=torch.float16, device=w.device)
    for r in range(0, N, rows):
        g = w[r:r + rows].float().view(-1, K // 32, 32)
        s = (g.abs().amax(2) / 127).clamp_min(1e-12).half()
        q[r:r + rows] = torch.round(g / s.float()[..., None]).clamp(-127, 127).to(torch.int8).view(-1, K)
        sc[r:r + rows] = s
    return q, sc.view(torch.int16)


class Segs:
    """gemv_seg の区切りの表（GPU のメモリー）。segs は (Lin, out, mode, Lin2, aux0, aux1) の並び。"""

    MODES = {"store": 0, "add": 1, "pair": 2, "conv": 3}

    def __init__(self, segs: list[tuple], device):
        K = segs[0][0].K
        rows, blocks = [], 0
        for lin, out, mode, lin2, aux0, aux1 in segs:
            assert lin.K == K and (lin2 is None or (lin2.K == K and lin2.N == lin.N))
            m = self.MODES[mode]
            tasks = lin.N if m == 2 else -(-lin.N // 2)
            nb = -(-tasks // 8)
            rows.append([int(_ptr(lin.w)), int(_ptr(lin.s)), int(_ptr(None if lin2 is None else lin2.w)),
                         int(_ptr(None if lin2 is None else lin2.s)), int(_ptr(out)), int(_ptr(aux0)), int(_ptr(aux1)),
                         lin.N, m | (lin.pol << 8), blocks])
            blocks += nb
        self.tab = torch.tensor(rows, dtype=torch.int64, device=device).view(-1).contiguous()
        self.n, self.K, self.blocks = len(segs), K, blocks
        self.keep = segs  # 出力などのテンソルを生かしておく


def _zc(norm) -> torch.Tensor:
    """ZeroCenteredRMSNorm の重み（1 + w、float32）"""
    return (1.0 + norm.weight.float()).contiguous()


def rope_tables(cap: int, device, theta: float, rotary_dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """[位置][cos 32 | sin 32]（float32）。言語モデル（backbone.run_layers）と塔（TowerAttention.rotate）の式のまま"""
    pos = torch.arange(cap, device=device)
    ex = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    f_bb = 1.0 / (theta ** (ex / rotary_dim))
    a_bb = pos.float()[:, None] * f_bb[None, :]
    a_tw = pos.float()[:, None] * theta ** (-ex / rotary_dim)
    return (torch.cat([a_bb.cos(), a_bb.sin()], 1).contiguous(), torch.cat([a_tw.cos(), a_tw.sin()], 1).contiguous())


class Kern:
    """コンパイルしたカーネルと起動の手続き。"""

    NAMES = ("gemv_seg", "gdn", "attn", "moe_gu", "moe_down", "embed_backbone", "rmsnorm_adv", "tower_prep", "sample_main",
             "sample_code")

    def __init__(self, wd: str):
        import cupy as cp

        self.cp = cp
        self.wd = wd
        opts = ("-std=c++14",) + {"bf16": (), "i8g32": ("-DWD_I8G32",), "f32": ("-DWD_F32", "-DACT_F32")}[wd]
        t0 = time.perf_counter()
        mod = cp.RawModule(code=_SRC, options=opts)
        self.k = {n: mod.get_function(n) for n in self.NAMES}
        self.compile_sec = time.perf_counter() - t0

    def __call__(self, name: str, grid: int, block: int, args: tuple, smem: int = 0) -> None:
        conv = tuple(_ptr(a) if (a is None or isinstance(a, torch.Tensor)) else a for a in args)
        self.k[name]((int(grid),), (int(block),), conv, shared_mem=int(smem))

    def gemv(self, src, nw, eps, xn_out, segs: Segs) -> None:
        self("gemv_seg", segs.blocks, 256, (src, nw, np.float32(eps), xn_out, segs.tab, np.int32(segs.n), np.int32(segs.K)),
             smem=segs.K * 4)


class Bufs:
    """1 フレームの途中の値（float32、GPU）"""

    def __init__(self, device):
        f = dict(dtype=torch.float32, device=device)
        z = lambda n: torch.zeros(n, **f)  # noqa: E731
        self.x, self.hid, self.xn = z(2048), z(2048), z(2048)
        self.qkv, self.conv, self.zg, self.b, self.a, self.y = z(9216), z(8192), z(4096), z(32), z(32), z(4096)
        self.router, self.sact, self.sg, self.act = z(256), z(512), z(4), z(8 * 512)
        self.ids = torch.zeros(8, dtype=torch.int32, device=device)
        self.wts = z(8)
        self.tr, self.txin, self.tqkv, self.tatt, self.tact = z(2048), z(4096), z(9216), z(4096), z(6144)
        self.lmain, self.lcode = z(4097), z(4096)


def backbone_layer_spec(layer, wd: str, B: Bufs, device, experts=None) -> dict:
    """言語モデルの 1 層（DecoderLayer）をカーネルの形にする。experts は FP8 のエキスパート（省けば layer のもの。
    float32 の層と突き合わせるときに、元の FP8 の層のものを渡す）。"""
    mlp = layer.mlp
    ex = mlp.experts if experts is None else experts
    assert ex.fp8, "エキスパートは FP8（公式の bsgemm-moe）のまま読む"
    se = mlp.shared_expert
    L = {"type": layer.layer_type, "n1": _zc(layer.input_layernorm), "n2": _zc(layer.post_attention_layernorm),
         "eps": float(layer.input_layernorm.eps)}
    if layer.layer_type == "full_attention":
        a = layer.self_attn
        L["in"] = Segs([(Lin(a.q_proj.weight, wd), B.qkv[:8192], "store", None, None, None),
                        (Lin(a.k_proj.weight, wd), B.qkv[8192:8704], "store", None, None, None),
                        (Lin(a.v_proj.weight, wd), B.qkv[8704:], "store", None, None, None)], device)
        L["qn"], L["kn"] = _zc(a.q_norm), _zc(a.k_norm)
        L["kc"], L["vc"] = a.k_cache[0], a.v_cache[0]
        L["o"] = Segs([(Lin(a.o_proj.weight, wd), B.x, "add", None, None, None)], device)
    else:
        g = layer.linear_attn
        L["in"] = Segs([(Lin(g.in_proj_qkv.weight, wd), B.conv, "conv", None, g.conv_state[0], g.conv1d.weight),
                        (Lin(g.in_proj_z.weight, wd), B.zg, "store", None, None, None),
                        (Lin(g.in_proj_b.weight, wd), B.b, "store", None, None, None),
                        (Lin(g.in_proj_a.weight, wd), B.a, "store", None, None, None)], device)
        assert g.conv_state.is_contiguous() and g.conv1d.weight.is_contiguous()
        L["A_log"], L["dt_bias"] = g.A_log.float().contiguous(), g.dt_bias.float().contiguous()
        L["gw"], L["geps"] = g.norm.weight.float().contiguous(), float(g.norm.eps)
        L["state"] = g.recurrent_state[0]
        assert L["state"].dtype == torch.float32 and L["state"].is_contiguous()
        L["o"] = Segs([(Lin(g.out_proj.weight, wd), B.x, "add", None, None, None)], device)
    L["moe_in"] = Segs([(Lin(mlp.gate.weight, wd), B.router, "store", None, None, None),
                        (Lin(se.gate_proj.weight, wd), B.sact, "pair", Lin(se.up_proj.weight, wd), None, None),
                        (Lin(mlp.shared_expert_gate.weight, wd), B.sg, "store", None, None, None)], device)
    L["w13"], L["s13"], L["w2"], L["s2"] = ex.w13_weight, ex.w13_scale_inv, ex.w2_weight, ex.w2_scale_inv
    L["sd"] = Lin(se.down_proj.weight.contiguous(), wd)  # K = 512（moe_down が読む）
    return L


def run_backbone_layer(k: Kern, L: dict, B: Bufs, pos, pos_add: int, rope: torch.Tensor, cap: int) -> None:
    """B.x（残差）に 1 層を流す。pos は位置（int64 のテンソル）、位置 = pos[0] + pos_add。"""
    if L["type"] == "full_attention":
        k.gemv(B.x, L["n1"], L["eps"], None, L["in"])
        k("attn", 16, 256, (B.qkv, np.int32(1), np.int32(1), L["qn"], L["kn"], np.float32(L["eps"]), rope, L["kc"], L["vc"],
                            pos, np.int32(pos_add), np.int32(cap), B.y), smem=cap * 4)
    else:
        k.gemv(B.x, L["n1"], L["eps"], None, L["in"])
        k("gdn", 32, 512, (B.conv, B.zg, B.b, B.a, L["A_log"], L["dt_bias"], L["gw"], np.float32(L["geps"]), L["state"], B.y))
    k.gemv(B.y, None, 0.0, None, L["o"])
    k.gemv(B.x, L["n2"], L["eps"], B.xn, L["moe_in"])
    k("moe_gu", 512, 256, (B.router, B.xn, L["w13"], L["s13"], B.act, B.ids, B.wts))
    k("moe_down", 256, 256, (B.act, B.ids, B.wts, L["w2"], L["s2"], B.sact, B.sg, L["sd"].w, L["sd"].s, B.x))


def tower_layer_spec(layer, wd: str, B: Bufs, device) -> dict:
    """塔の 1 層（TowerLayer）をカーネルの形にする。"""
    emb = layer.semantic_embedding if layer.shared else layer.codebook_embedding
    tl = layer.transformer_layer
    a, ff = tl.attention, tl.feed_forward
    return {
        "emb": emb.weight, "enw": _zc(layer.enorm), "hnw": _zc(layer.hnorm), "fnw": _zc(layer.final_layernorm),
        "eps": float(layer.enorm.eps),
        "eh": Segs([(Lin(layer.eh_proj.weight, wd), B.tr, "store", None, None, None)], device),
        "an": _zc(tl.attention_norm), "fn": _zc(tl.ffn_norm),
        "qkv": Segs([(Lin(a.wqkv.weight, wd), B.tqkv, "store", None, None, None)], device),
        "qn": _zc(a.q_norm), "kn": _zc(a.k_norm), "kc": a.k_cache[0], "vc": a.v_cache[0],
        "wo": Segs([(Lin(a.wo.weight, wd), B.tr, "add", None, None, None)], device),
        "w13": Segs([(Lin(ff.w1.weight, wd), B.tact, "pair", Lin(ff.w3.weight, wd), None, None)], device),
        "w2": Segs([(Lin(ff.w2.weight, wd), B.tr, "add", None, None, None)], device),
    }


def run_tower_layer(k: Kern, T: dict, B: Bufs, src, fnw_prev, frame_out, code_idx: int, cb, frame_in, pos, rope,
                    cap: int) -> None:
    """塔の 1 層。入力の隠れ状態は src（fnw_prev があれば前の層の final_layernorm をかける）。結果は B.tr（final_layernorm の前）。"""
    eps = T["eps"]
    k("tower_prep", 1, 1024, (src, fnw_prev, T["hnw"], T["emb"], T["enw"], frame_out, np.int32(code_idx), cb, frame_in,
                              np.float32(eps), B.txin, np.int32(SEM_BEGIN), np.int32(SEM_END)))
    k.gemv(B.txin, None, 0.0, None, T["eh"])
    k.gemv(B.tr, T["an"], eps, None, T["qkv"])
    k("attn", 16, 256, (B.tqkv, np.int32(0), np.int32(0), T["qn"], T["kn"], np.float32(eps), rope, T["kc"], T["vc"], pos,
                        np.int32(0), np.int32(cap), B.tatt), smem=cap * 4)
    k.gemv(B.tatt, None, 0.0, None, T["wo"])
    k.gemv(B.tr, T["fn"], eps, None, T["w13"])
    k.gemv(B.tact, None, 0.0, None, T["w2"])


def _common_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    """2 つのプロンプト（[rows, 11]、CPU）の頭から同じ行の数。"""
    n = min(len(a), len(b))
    same = (a[:n] == b[:n]).all(1)
    return n if bool(same.all()) else int((~same).nonzero()[0, 0])


def drop_unused_codec_masks(codec) -> int:
    """コーデックの WindowLimitedTransformer が持つ 32768 × 32768 の因果マスク（1 つ 1 GiB、3 つ）を空にし、空けたバイト数を返す。

    WindowLimitedTransformer.forward は毎回マスクを作って Transformer.forward に渡すので、このバッファーは読まれない
    （Transformer.forward が causal_mask を使うのは、マスクを渡されないときだけ）。空けたメモリーに int8 の重みを置く。"""
    from fish_speech.models.dac.modded_dac import WindowLimitedTransformer

    freed = 0
    for m in codec.modules():
        mask = getattr(m, "causal_mask", None)
        if mask is None or mask.numel() == 0:
            continue
        assert type(m) is WindowLimitedTransformer, f"{type(m).__name__} は causal_mask を読むことがある"
        freed += mask.numel() * mask.element_size()
        m.causal_mask = torch.zeros(0, dtype=mask.dtype, device=mask.device)
    torch.cuda.empty_cache()
    return freed


class Drama3Engine:
    """公式の FishOmniModel（bf16、エキスパートは FP8、CUDA の 1 枚）の 1 フレームを、自作のカーネルで動かす。

    generate(model, prompt, ...) は公式の generate_drama3 と同じ約束（プロンプトの行 [rows, 11] を受け取り、EOS の前までの
    フレーム [frames, 11]（CPU）と終わり方を返す）。fish-speech の generate_long_drama3 の generate_drama3 をこれに差し替えると、
    文の区切り方と、前の区切りをプロンプトに入れるやり方は公式のまま使える（drama3/bench.py --engine kernels）。
    最初の入力は公式のモジュール（embed_backbone、backbone、tower.prefill。128 行ずつ）で流し、キャッシュは公式のテンソルを使う。
    1 フレームは 1 つの CUDA Graph（位置は GPU のメモリーから読むので、どの位置でも同じグラフ）。CPU は 1 フレーム遅れて
    EOS を見る（フレーム n の結果を待つ前にフレーム n+1 を流す。EOS のあとの 1 フレームは捨てる）。
    """

    def __init__(self, model, wd: str = "bf16", graphs: bool = True, prefill_graphs: int = 0, prefix_cache: bool = False):
        from fish_speech.models.text2semantic.fish_omni.generate import (DRAMA3_GRAPH_CAPACITY, ensure_drama3_caches)

        self.model = model
        bb, tw = model.backbone, model.audio_decoder
        dev = tw.device
        assert dev.type == "cuda" and all(p.device == dev for p in bb.parameters()), "1 枚の GPU に載せたモデルだけを扱う"
        assert torch.cuda.get_device_capability(dev) >= (8, 9), "FP8 の変換（cvt.rn.f16x2.e4m3x2）は sm_89 以降"
        cfg = model.config
        assert (cfg.semantic_start_token_id, cfg.semantic_end_token_id, cfg.eos_token_id) == (SEM_BEGIN, SEM_END, EOS)
        assert cfg.audio_decoder.num_codebooks == 10 and cfg.audio_decoder.vocab_size == 4096
        self.cap = DRAMA3_GRAPH_CAPACITY
        ensure_drama3_caches(model, self.cap)  # 公式のキャッシュ（この大きさから変えない。グラフがアドレスを持つ）
        self.device, self.wd, self.use_graphs = dev, wd, graphs
        t0 = time.perf_counter()
        self.k = Kern(wd)
        B = self.B = Bufs(dev)
        with torch.no_grad():
            self.layers = [backbone_layer_spec(layer, wd, B, dev) for layer in bb.model.layers]
            self.fnorm, self.feps = _zc(bb.model.norm), float(bb.model.norm.eps)
            w = bb.lm_head.weight
            # main token の候補（eos と semantic の帯）の行だけを並べて写す（16.8 MB）
            cand = torch.cat([w[EOS:EOS + 1], w[SEM_BEGIN:SEM_END + 1]]).contiguous()
            self.lm = Segs([(Lin(cand, wd), B.lmain, "store", None, None, None)], dev)
            self.temb, self.cb = bb.model.embed_tokens.weight, tw.codebook_embeddings.weight
            self.tshared = [tower_layer_spec(layer, wd, B, dev) for layer in tw.shared_layers]
            self.tres = [tower_layer_spec(layer, wd, B, dev) for layer in tw.residual_layers]
            self.heads = [Segs([(Lin(h.weight, wd), B.lcode, "store", None, None, None)], dev) for h in tw.codebook_heads]
            assert abs(cfg.text.rope_theta - cfg.audio_decoder.rope_base) < 1e-3
            assert cfg.text.rotary_dim == 64 and cfg.audio_decoder.rotary_dim == 64
            self.rope_bb, self.rope_tw = rope_tables(self.cap, dev, float(cfg.text.rope_theta))
        i64 = dict(dtype=torch.int64, device=dev)
        self.pos, self.step = torch.zeros(1, **i64), torch.zeros(1, **i64)
        self.frame_in, self.frame_out = torch.zeros(WIDTH, **i64), torch.zeros(WIDTH, **i64)
        self.history = torch.full((16,), -1, **i64)
        self.frames = torch.zeros((self.cap + 2, WIDTH), **i64)
        self.seed, self.counter = torch.zeros(1, **i64), torch.zeros(1, **i64)
        self.params = torch.tensor([1.0, 0.9, 30.0], dtype=torch.float32, device=dev)
        self.bad = torch.zeros(1, dtype=torch.int32, device=dev)
        # CPU が読む main token（ページ固定のメモリー。GPU のカーネルが直接書く）
        self.host_main = torch.full((self.cap + 2,), -1, dtype=torch.int64).pin_memory()
        self.graph = None
        # 最初の入力の CUDA Graph（公式のモジュールをそのまま記録する。(start, end, 塔の end) ごとに 1 つ）
        self.pf_frames = torch.zeros((self.cap + 1, WIDTH), **i64)
        self.pf_pos = torch.arange(self.cap + 1, **i64)
        self.pf_hidden = torch.zeros(cfg.text.hidden_size, dtype=bb.dtype, device=dev)
        self.pf_graphs: OrderedDict[tuple[int, int, int], torch.cuda.CUDAGraph] = OrderedDict()
        self.pf_pool = None
        self.pf_capacity, self.pf_captures = prefill_graphs, 0
        # 区切りごとに流した回数。2 回目に出た区切りから記録する（長い文の 2 区切り目からは、前の区切りがプロンプトに
        # 入って長さが毎回変わる。1 回しか出ない区切りを記録すると、記録（約 0.5 秒）が eager（0.1〜0.3 秒）より遅い）
        self.pf_seen: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        # 参照の行を使い回す（prefill を見る）
        self.prefix_cache, self.prefix_min = prefix_cache, 64
        self.prefix, self.last_prompt, self.gdn_snap = None, None, None
        self.setup_sec = time.perf_counter() - t0

    # --- 1 フレーム -----------------------------------------------------------------------------------------------
    def _frame_launches(self, hook=None) -> None:
        """1 フレームのカーネルを並べる。hook（確かめ用。グラフなしのときだけ）は、引いた直後に hook("main", 0) と
        hook("code", i) で呼ばれ、ロジット（B.lmain、B.lcode）を写したり、frame_out を書き換えたり（教師強制）できる。"""
        k, B, cap = self.k, self.B, self.cap
        k.gemv(B.hid, None, 0.0, None, self.lm)
        k("sample_main", 1, 1024, (B.lmain, np.int32(4097), self.params, self.seed, self.counter, self.history,
                                   self.frame_out, self.host_main, self.step, self.bad, np.int32(EOS), np.int32(SEM_BEGIN),
                                   np.int32(SEM_END)))
        if hook:
            hook("main", 0)
        src, fprev = B.hid, None
        for i, T in enumerate(self.tshared):
            run_tower_layer(k, T, B, src, fprev, self.frame_out, 1, self.cb if i == 0 else None, self.frame_in, self.pos,
                            self.rope_tw, cap)
            src, fprev = B.tr, T["fnw"]
        for i, T in enumerate(self.tres):
            run_tower_layer(k, T, B, src, fprev, self.frame_out, 1 + i, None, None, self.pos, self.rope_tw, cap)
            src, fprev = B.tr, T["fnw"]
            k.gemv(B.tr, T["fnw"], T["eps"], None, self.heads[i])
            k("sample_code", 1, 1024, (B.lcode, self.params, self.seed, self.counter, self.frame_out, np.int32(2 + i),
                                       np.int32(int(i == len(self.tres) - 1)), self.frames, self.step, self.bad))
            if hook:
                hook("code", i)
        k("embed_backbone", 8, 256, (self.temb, self.cb, self.frame_out, self.frame_in, B.x, np.int32(2048),
                                     np.int32(SEM_BEGIN), np.int32(SEM_END)))
        for L in self.layers:
            run_backbone_layer(k, L, B, self.pos, 1, self.rope_bb, cap)
        k("rmsnorm_adv", 1, 1024, (B.x, self.fnorm, np.float32(self.feps), B.hid, self.pos))

    def frame(self, hook=None) -> None:
        """1 フレームを今の torch のストリームに流す（グラフがあれば再生する。hook はグラフなしのときだけ）。"""
        if self.graph is not None:
            assert hook is None
            self.graph.replay()
            return
        with self.k.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            self._frame_launches(hook)

    def capture(self) -> float:
        """1 フレームを CUDA Graph に記録する（その前に 1 回流してカーネルを読み込む。状態は次の要求の最初の入力で戻る）。"""
        t0 = time.perf_counter()
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with self.k.cp.cuda.ExternalStream(s.cuda_stream):
                self._frame_launches()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with self.k.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                self._frame_launches()
        torch.cuda.synchronize()
        self.graph = g
        return time.perf_counter() - t0

    # --- 最初の入力（公式のモジュール）---------------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, prompt: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """最初の入力を流し、最後の行の隠れ状態（最後の RMSNorm のあと、bf16）を返す。

        計算は公式の generate_drama3 の最初の入力と同じ（128 行ずつ、塔のキャッシュは行 p に p+1 のフレームを入れる）。
        prefill_graphs > 0 なら、区切り（start, end）ごとに CUDA Graph に記録して流す（eager と同じ値）。初めての区切りは
        eager で流し、2 回目に出たときに記録してから流し直す。グラフは新しい順に prefill_graphs 個まで持つ。
        prefix_cache なら、前の要求と頭の prefix_min 行以上が同じとき（同じ参照を使う要求）、共通の行の 1 つ手前 B までを
        区切って流し、B での Gated DeltaNet の状態を写しておく。次の要求の頭が同じなら、状態を戻して B から先だけを流す
        （B より前の KV と塔の KV は、キャッシュに残っている）。区切り方が公式と変わるので、値は公式と少し違う。"""
        rows = len(prompt)
        assert 1 <= rows <= len(self.pf_frames) - 1
        prompt = prompt.cpu()
        self.pf_frames[:rows].copy_(prompt, non_blocking=False)
        P, snap_after = self.prefix, None
        if P is not None and rows > len(P) and torch.equal(prompt[:len(P)], P):
            spans = self._spans(len(P) - 1, rows, chunk)
            reuse = True
        else:
            reuse = False
            lcp = _common_rows(self.last_prompt, prompt) if self.prefix_cache and self.last_prompt is not None else 0
            lcp = min(lcp, rows - 1)
            self.prefix = None  # この要求が B より前の KV を書き換える
            if lcp >= self.prefix_min:
                head = self._spans(0, lcp - 1, chunk)
                spans, snap_after = head + self._spans(lcp - 1, rows, chunk), len(head)
                new_prefix = prompt[:lcp].clone()
            else:
                spans = self._spans(0, rows, chunk)
        self.last_prompt = prompt
        keys = [(st, en, min(en, rows - 1)) for st, en in spans]
        if self.pf_capacity > 0:
            for key in keys:
                if key in self.pf_graphs:
                    self.pf_graphs.move_to_end(key)
                elif self.pf_seen.get(key, 0) >= 1:
                    self._capture_prefill(key)  # 記録の前に流す 2 回が状態を進めるので、このあと初めから流し直す
            for key in keys:
                self.pf_seen[key] = self.pf_seen.pop(key, 0) + 1
            while len(self.pf_seen) > 4096:
                self.pf_seen.popitem(last=False)
            old = [k for k in self.pf_graphs if k not in keys]  # 古い順。この要求で使うものは残す
            for k in old[: max(0, len(self.pf_graphs) - self.pf_capacity)]:
                del self.pf_graphs[k]
        if reuse:
            self._restore_gdn()
        for j, key in enumerate(keys):
            if j == snap_after:
                self._save_gdn()
            g = self.pf_graphs.get(key) if self.pf_capacity > 0 else None
            if g is not None:
                g.replay()
            else:
                self._prefill_span(*key)
        if snap_after is not None:
            self.prefix = new_prefix
        return self.pf_hidden

    @staticmethod
    def _spans(a: int, b: int, chunk: int) -> list[tuple[int, int]]:
        return [(st, min(st + chunk, b)) for st in range(a, b, chunk)]

    def _gdn_layers(self):
        return [layer.linear_attn for layer in self.model.backbone.model.layers if layer.layer_type != "full_attention"]

    def _save_gdn(self) -> None:
        if self.gdn_snap is None:
            self.gdn_snap = [(g.recurrent_state.clone(), g.conv_state.clone()) for g in self._gdn_layers()]
        else:
            for g, (r, c) in zip(self._gdn_layers(), self.gdn_snap):
                r.copy_(g.recurrent_state)
                c.copy_(g.conv_state)

    def _restore_gdn(self) -> None:
        for g, (r, c) in zip(self._gdn_layers(), self.gdn_snap):
            g.recurrent_state.copy_(r)
            g.conv_state.copy_(c)

    def _tower_attentions(self):
        tw = self.model.audio_decoder
        for layer in (*tw.shared_layers, *tw.residual_layers):
            yield layer.transformer_layer.attention

    def _prefill_span(self, start: int, end: int, paired_end: int) -> None:
        """pf_frames の行 start..end-1 を公式のモジュールで流す（公式の generate_drama3 の 1 区切りと同じ計算）。
        CPU と同期しないよう、backbone.forward の代わりに run_layers を呼び（forward は位置を int にする）、塔の注意には
        公式のグラフと同じく static_kv_len を渡す。最後の行の隠れ状態を pf_hidden に書く。"""
        bb, tw = self.model.backbone, self.model.audio_decoder
        part = self.pf_frames[start:end]
        emb = tw.embed_backbone(part, bb.embed(part[:, 0]))
        hidden = bb.run_layers(emb[None].to(bb.residual_dtype), self.pf_pos[start:end], kv_len=end,
                               continue_state=start > 0, use_cache=True)[0]
        if paired_end > start:
            for a in self._tower_attentions():
                a.static_kv_len = paired_end
            try:
                tw.prefill(hidden[: paired_end - start], self.pf_pos[start:paired_end], self.pf_frames[start:paired_end],
                           self.pf_frames[start + 1: paired_end + 1])
            finally:
                for a in self._tower_attentions():
                    a.static_kv_len = None
        self.pf_hidden.copy_(hidden[-1])

    def _capture_prefill(self, key: tuple[int, int, int]) -> None:
        if self.pf_pool is None:
            self.pf_pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # Triton の読み込みなど
                self._prefill_span(*key)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pf_pool):
            self._prefill_span(*key)
        self.pf_graphs[key] = g
        self.pf_captures += 1

    def start(self, hidden: torch.Tensor, prompt: torch.Tensor, temperature: float, top_k: int, top_p: float) -> None:
        """最初の入力のあとの状態を入れる（最後の行の隠れ状態、位置、フレーム、直近 16 個の main token、サンプリングの設定）。"""
        rows = len(prompt)
        self.B.hid.copy_(hidden.float())
        self.pos.fill_(rows - 1)
        self.step.zero_()
        self.frame_in.copy_(prompt[-1].to(self.device))
        window = prompt[-16:, 0].to(self.device)
        self.history.fill_(-1)
        self.history[16 - len(window):].copy_(window)
        self.params.copy_(torch.tensor([float(temperature), float(top_p), float(top_k)], dtype=torch.float32))
        self.bad.zero_()
        # 自前の乱数の鍵は torch の CPU の乱数から引く（set_seed で同じ列になり、続けて呼べば毎回変わる）
        self.seed.fill_(int(torch.randint(0, 2**62, (1,)).item()))
        self.counter.zero_()

    @torch.inference_mode()
    def generate(self, model, prompt: torch.Tensor, *, max_new_tokens: int, temperature: float = 0.7, top_k: int = 30,
                 top_p: float = 0.7, epsilon_cutoff: float | None = None, prefill_chunk_size: int = 128):
        """公式の generate_drama3 と同じ約束。reference_sampling（Fish Audio のサンプラー）と epsilon_cutoff には対応しない。"""
        from fish_speech.models.text2semantic.errors import InvalidInputError

        assert model is self.model
        if model.audio_decoder.reference_sampling is not None or epsilon_cutoff:
            raise InvalidInputError("このエンジンは既定のサンプラー（reference_sampling なし）だけに対応する")
        if top_k < 1:
            raise InvalidInputError(f"top_k must be at least 1; got {top_k}")
        if top_k > 64:
            raise InvalidInputError("このエンジンの top_k は 64 まで")
        rows = len(prompt)
        budget = min(max_new_tokens, self.cap - rows - 1)
        if budget < 1:
            raise ValueError(f"max_new_tokens={max_new_tokens} does not fit after {rows} prompt rows")
        if self.use_graphs and self.graph is None:
            self.capture()  # 記録のときに流す 1 フレームが変えたキャッシュと状態は、このあとの最初の入力で戻る
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = self.prefill(prompt, prefill_chunk_size)
        self.start(hidden, prompt, temperature, top_k, top_p)
        torch.cuda.synchronize()
        from loguru import logger

        logger.info(f"Drama 3 prefill: {rows} rows in {time.perf_counter() - t0:.4f}s")  # 公式と同じ形（drama3/bench.py が拾う）
        hm = self.host_main.numpy()
        ev = [torch.cuda.Event(), torch.cuda.Event()]
        launched, n, finish = 0, 0, "length"

        def launch() -> None:
            nonlocal launched
            self.frame()
            ev[launched % 2].record()
            launched += 1

        launch()
        while n < budget:
            if launched < budget:
                launch()
            ev[n % 2].synchronize()
            if int(hm[n]) == EOS:
                finish = "eos"
                break
            n += 1
        torch.cuda.synchronize()
        if int(self.bad.item()):
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        return self.frames[:n].cpu(), finish
