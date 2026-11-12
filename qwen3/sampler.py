"""1 回のサンプリング（processor、温度、top-k、引く）を 1 つの CUDA カーネルで行う。qwen3/kernels.py の sampler="fused" が使う。

torch の演算で書くと、1 回に約 10 個のカーネルが動き（割り算、topk、masked_fill、softmax、exponential_、argmax、コピー）、
1 フレーム（16 回）で約 0.5 ms かかる。このカーネルは 1 ブロック（1024 スレッド）で 1 回分を済ませる。
  1. logits を読み、言語モデルのときは公式の processor をかける（繰り返しの罰、抑える番号、min_new_tokens の EOS）
  2. 温度で割り、上から top_k 番目の値を基数選択（8 ビットずつ 4 回）で求める
  3. その値以上の番号の中で、argmax(l_i - log E_i)（E_i ~ Exp(1)）を選ぶ。softmax して p_i / E_i の最大を選ぶのと同じ
乱数は自前の Philox4x32-10（鍵 = seed、カウンター = (番号, 引いた回数)）。torch の乱数とは別の数になるので、同じシードでも
公式とは違う列になる（分布は同じ）。公式とコードの列を突き合わせるときは、torch の演算の経路（sampler="torch"）を使う。
引いた回数のカウンターは GPU のメモリーに置き、カーネルが 1 つ進める（CUDA Graph を再生するたびに違う数になる）。
"""

from __future__ import annotations

import numpy as np
import torch

SRC = r'''
#define SB 1024
static __device__ __forceinline__ unsigned int fkey(float f){  // 大小の順を保つ符号なし整数
  unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static __device__ __forceinline__ uint4 philox(uint4 c, uint2 k){
  #pragma unroll
  for (int i = 0; i < 10; i++){
    unsigned int hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    unsigned int hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u; k.y += 0xBB67AE85u;
  }
  return c;
}

// logits: [V]（float32）。out[0] に選んだ番号を書く。talker が 1 なら言語モデルの後始末もする
// （id0 = 番号、seen[番号] = 1、ngen += 1、pos += 1）。seen・suppress は bool（1 バイト）。
extern "C" __global__ void sample(const float* __restrict__ logits, int V, float inv_temp, int top_k, int do_sample,
                                  unsigned char* __restrict__ seen, float rep_pen, const unsigned char* __restrict__ suppress,
                                  long long* __restrict__ ngen, int min_new, int eos,
                                  const unsigned long long* __restrict__ seed, unsigned long long* __restrict__ counter,
                                  long long* __restrict__ out, int talker, long long* __restrict__ id0,
                                  long long* __restrict__ pos){
  __shared__ unsigned int hist[256];
  __shared__ unsigned int s_prefix, s_need;
  __shared__ float s_best[SB / 32];
  __shared__ int s_idx[SB / 32];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  #define PER 4
  float l[PER];
  const bool block_eos = talker && ngen[0] < min_new;
  #pragma unroll
  for (int j = 0; j < PER; j++){
    int i = t + j * SB;
    float v = -__int_as_float(0x7f800000);
    if (i < V){
      v = logits[i];
      if (talker){
        if (seen && seen[i]) v = v < 0.f ? v * rep_pen : v / rep_pen;
        if (suppress && suppress[i]) v = -__int_as_float(0x7f800000);
        if (block_eos && i == eos) v = -__int_as_float(0x7f800000);
      }
      if (do_sample) v *= inv_temp;
    }
    l[j] = v;
  }
  // top_k 番目の値（の順序キー）を求める
  unsigned int thr = 0u;
  if (do_sample && top_k > 0 && top_k < V){
    if (t == 0){ s_prefix = 0u; s_need = (unsigned int)top_k; }
    for (int pass = 0; pass < 4; pass++){
      const int shift = 24 - 8 * pass;
      if (t < 256) hist[t] = 0u;
      __syncthreads();
      const unsigned int prefix = s_prefix, need = s_need;
      const unsigned int hmask = pass ? (0xffffffffu << (shift + 8)) : 0u;
      #pragma unroll
      for (int j = 0; j < PER; j++){
        int i = t + j * SB;
        if (i < V){
          unsigned int k = fkey(l[j]);
          if ((k & hmask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
      }
      __syncthreads();
      if (w == 0){
        // 上の桁から数えて、need 個に届く桁を探す（lane が 8 個ずつ、上の桁から）
        unsigned int c[8], s = 0u;
        #pragma unroll
        for (int q = 0; q < 8; q++){ c[q] = hist[255 - (lane * 8 + q)]; s += c[q]; }
        unsigned int inc = s;  // lane までの累積（lane を含む）
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1){ unsigned int y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
        unsigned int before = inc - s;
        bool mine = before < need && inc >= need;
        if (mine){
          unsigned int acc = before;
          #pragma unroll
          for (int q = 0; q < 8; q++){
            if (acc + c[q] >= need){
              unsigned int digit = 255u - (unsigned int)(lane * 8 + q);
              s_prefix = prefix | (digit << shift);
              s_need = need - acc;
              break;
            }
            acc += c[q];
          }
        }
      }
      __syncthreads();
    }
    thr = s_prefix;
  }
  // 引く（do_sample が 0 なら argmax）
  float best = -__int_as_float(0x7f800000);
  int bi = 0x7fffffff;
  const unsigned long long sd = seed[0], cnt = counter[0];
  const uint2 key = make_uint2((unsigned int)sd, (unsigned int)(sd >> 32));
  #pragma unroll
  for (int j = 0; j < PER; j++){
    int i = t + j * SB;
    if (i >= V) continue;
    float v = l[j];
    if (do_sample){
      if (fkey(v) < thr || v == -__int_as_float(0x7f800000)) continue;
      uint4 r = philox(make_uint4((unsigned int)i, (unsigned int)cnt, (unsigned int)(cnt >> 32), 0x51u), key);
      float u = ((float)(r.x >> 8) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1)
      v = v - logf(-logf(u));
    }
    if (v > best || (v == best && i < bi)){ best = v; bi = i; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1){
    float ob = __shfl_xor_sync(0xffffffffu, best, o);
    int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
  }
  if (lane == 0){ s_best[w] = best; s_idx[w] = bi; }
  __syncthreads();
  if (w == 0){
    best = s_best[lane]; bi = s_idx[lane];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1){
      float ob = __shfl_xor_sync(0xffffffffu, best, o);
      int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
    }
    if (lane == 0){
      if (bi == 0x7fffffff) bi = 0;
      out[0] = bi;
      if (do_sample) counter[0] = cnt + 1ull;
      if (talker){
        if (id0) id0[0] = bi;
        if (seen) seen[bi] = 1;
        ngen[0] += 1;
        if (pos) pos[0] += 1;
      }
    }
  }
}
'''

MAX_V = 4096


class FusedSampler:
    """sample カーネルを起動する。seed と counter は GPU のメモリー（int64、1 個）。"""

    def __init__(self, device):
        import cupy as cp

        self.k = cp.RawModule(code=SRC, options=("-std=c++14",)).get_function("sample")
        self.seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)

    def set_seed(self, seed: int) -> None:
        self.seed.fill_(int(seed) & 0x7FFFFFFFFFFFFFFF)
        self.counter.zero_()

    def __call__(self, logits: torch.Tensor, out: torch.Tensor | int, do_sample: bool, temperature: float, top_k: int,
                 talker: dict | None = None) -> None:
        """logits（float32、V 個、連続）から 1 つ引き、out（int64 のテンソルか、そのアドレス）に書く。
        talker には言語モデルのときだけ seen、rep_pen、suppress、ngen、min_new、eos、id0、pos を渡す。"""
        V = logits.numel()
        assert V <= MAX_V and logits.dtype == torch.float32 and logits.is_contiguous()
        p = lambda t: np.uintp(0 if t is None else (t if isinstance(t, int) else t.data_ptr()))  # noqa: E731
        tk = talker or {}
        args = (p(logits), np.int32(V), np.float32(1.0 / temperature if do_sample else 1.0), np.int32(top_k or 0),
                np.int32(int(do_sample)), p(tk.get("seen")), np.float32(tk.get("rep_pen", 1.0)), p(tk.get("suppress")),
                p(tk.get("ngen")), np.int32(tk.get("min_new", 0)), np.int32(tk.get("eos", -1)), p(self.seed),
                p(self.counter), p(out), np.int32(int(talker is not None)), p(tk.get("id0")), p(tk.get("pos")))
        self.k((1,), (1024,), args)
