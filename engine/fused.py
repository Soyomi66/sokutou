"""DiT の小さな処理を自作の CUDA カーネルにまとめたエンジン（engine/bucket.py の BucketSampler を元にする）。

BucketSampler の 1 ステップは、行列積と attention のほかに、正規化・RoPE・連結・ゲートなどの小さなカーネルを
約 400 個起動する。1 個の計算は数マイクロ秒で、ほとんどが起動の固定費になっている。FusedCore は、
行列積（cuBLAS）と attention（PyTorch）のあいだの処理を、1 層あたり 5 個のカーネルにまとめる。

  1. res_adaln : 残差の足し込み（h_res += gate × o）と、次の AdaLN（正規化 → scale・shift → 行列積の型）
  2. qkv       : q と k の正規化・重み・RoPE、v の並べ替え、条件の K・V の連結（attention の並びで書く）
  3. gate      : attention の出力の並べ替えと、sigmoid(gate) の掛け算
  4. swiglu    : silu(a) × b
  （1 は attention の前と MLP の前の 2 回）

計算の式は engine/fast.py の DitCore と同じ。違いは次のとおり。
- 正規化の 2 乗和を double で足す（足す順序が変わるため）。
- BF16 のとき、sigmoid・silu とその掛け算を fp32 で計算してから 1 回だけ BF16 に丸める
  （DitCore は BF16 のまま計算するので、丸めが 2 回入る）。
- attention のマスクは、起動のたびに真偽値から作らず、足し込む値（0 か −inf）を形ごとに持つ。

カーネルは CuPy の RawModule（NVRTC）でコンパイルする。C コンパイラーは要らないが、cupy が要る
（docker/Dockerfile の dev の段のイメージ、sokutou/dev:7ed4374）。CUDA Graph の記録では、PyTorch の今のストリームで起動する。
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from engine.bucket import BucketSampler, _Shape
from engine.fast import DitCore, Voice, matmul_tf32, patch_runtime

_SRC = r'''
#ifdef WD_BF16
typedef unsigned short wd_t;
static __device__ __forceinline__ float LD(wd_t v){ return __uint_as_float(((unsigned int)v) << 16); }
static __device__ __forceinline__ wd_t ST(float f){  // 偶数丸め（PyTorch の BF16 への変換と同じ）
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (wd_t)(u >> 16);
}
#else
typedef float wd_t;
static __device__ __forceinline__ float LD(wd_t v){ return v; }
static __device__ __forceinline__ wd_t ST(float f){ return f; }
#endif

// 行列積に入れる値（adaln・res_adaln・gate・swiglu の出力）の型。OUT_E4M3 なら FP8（e4m3）で書く（engine/fp8.py）。
#ifdef OUT_E4M3
typedef unsigned char ot_t;
// fp32 → e4m3（符号 1、指数 4、仮数 3 ビット。最大 448、NaN は 0x7f）。偶数丸め。448 を超える値は ±448 にする。
static __device__ __forceinline__ ot_t E4(float f){
  unsigned int u = __float_as_uint(f);
  unsigned int sign = (u >> 24) & 0x80u;
  u &= 0x7fffffffu;
  if (u > 0x7f800000u) return 0x7f;
  float a = __uint_as_float(u);
  if (a >= 448.0f) return (ot_t)(sign | 0x7eu);
  if (a < 0.015625f) return (ot_t)(sign | (unsigned int)rintf(a * 512.0f));  // 2^-6 未満は 2^-9 刻み
  u += 0x7ffffu + ((u >> 20) & 1u);
  return (ot_t)(sign | (((u >> 23) - 120u) << 3) | ((u >> 20) & 7u));
}
#define STO(f, s) E4((f) * (s))
extern "C" __global__ void e4m3_encode(const float* x, ot_t* out, int n){  // 確かめ用
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = E4(x[i]);
}
#else
typedef wd_t ot_t;
#define STO(f, s) ST(f)
#endif

#define NT 256  // res_adaln・gate・swiglu の 1 ブロックのスレッド数

// ブロックの全スレッドの s の和。warp の中はシャッフルで足し、warp ごとの和を共有メモリーで集める。
static __device__ __forceinline__ double block_sum(double* red, double s){
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  double t = 0.0;
  for (int i = 0; i < NT / 32; i++) t += red[i];
  return t;
}

// 1 ブロック = 1 行。h = 行列積の型(x × rsqrt(mean(x²) + eps) × scale1 + shift)
// os は FP8 で書くときに掛ける倍率（OUT_E4M3 のときだけ使う。ほかのカーネルも同じ）。
extern "C" __global__ void adaln(const float* x, const float* scale1, const float* shift, ot_t* out,
                                 int D, float inv_dim, float eps, float os){
  __shared__ double red[NT / 32];
  const float* row = x + (size_t)blockIdx.x * D;
  double s = 0.0;
  for (int j = threadIdx.x; j < D; j += NT){ float v = row[j]; s += (double)v * v; }
  float r = rsqrtf((float)(block_sum(red, s) * inv_dim) + eps);
  ot_t* o = out + (size_t)blockIdx.x * D;
  for (int j = threadIdx.x; j < D; j += NT) o[j] = STO(row[j] * r * scale1[j] + shift[j], os);
}

// 1 ブロック = 1 行。x += g × o をその場で行い、続けて adaln と同じ計算をする。
extern "C" __global__ void res_adaln(float* x, const wd_t* o, const float* g, const float* scale1, const float* shift,
                                     ot_t* out, int D, float inv_dim, float eps, float os){
  __shared__ double red[NT / 32];
  float* row = x + (size_t)blockIdx.x * D;
  const wd_t* orow = o + (size_t)blockIdx.x * D;
  double s = 0.0;
  for (int j = threadIdx.x; j < D; j += NT){
    float v = row[j] + g[j] * LD(orow[j]);
    row[j] = v;
    s += (double)v * v;
  }
  float r = rsqrtf((float)(block_sum(red, s) * inv_dim) + eps);
  ot_t* h = out + (size_t)blockIdx.x * D;
  for (int j = threadIdx.x; j < D; j += NT) h[j] = STO(row[j] * r * scale1[j] + shift[j], os);
}

// 1 ブロック = (b, キーの位置 p)。スレッド数は 2 × heads × 8。
// p < L: o4 の行 (b, p) の q・k を head ごとに正規化して重みを掛け、前半の head に RoPE を掛けて書く。v はそのまま書く。
// p >= L: 条件の K・V（ctx、バッチで共通）を書く。出力は attention の並び (B, heads, 長さ, hd)。
extern "C" __global__ void qkv(const wd_t* o4, const float* qk_w, const float* freqs, const wd_t* ctx_k, const wd_t* ctx_v,
                               wd_t* q, wd_t* k, wd_t* v, int L, int S, int H, int HD, float inv_hd, float eps){
  __shared__ double part[QKV_NT];  // QKV_NT = 2 × heads × 8（起動するスレッドの数。FusedCore が -D で渡す）
  __shared__ float rs[QKV_NT / 8];
  const int D = H * HD, K = L + S;
  const int b = blockIdx.x / K, p = blockIdx.x % K;
  const int nt = blockDim.x;
  if (p >= L){
    const wd_t* ck = ctx_k + (size_t)(p - L) * D;
    const wd_t* cv = ctx_v + (size_t)(p - L) * D;
    for (int j = threadIdx.x; j < D; j += nt){
      int h = j / HD, e = j % HD;
      size_t dst = ((size_t)(b * H + h) * K + p) * HD + e;
      k[dst] = ck[j];
      v[dst] = cv[j];
    }
    return;
  }
  const wd_t* row = o4 + ((size_t)b * L + p) * 4 * D;
  {  // 2 × heads 個の正規化の係数。8 スレッドで 1 個。
    int grp = threadIdx.x >> 3, sub = threadIdx.x & 7;
    const wd_t* src = row + (size_t)grp * HD;
    double s = 0.0;
    for (int e = sub; e < HD; e += 8){ float t = LD(src[e]); s += (double)t * t; }
    part[threadIdx.x] = s;
    __syncthreads();
    if (sub == 0){
      for (int i = 1; i < 8; i++) s += part[threadIdx.x + i];
      rs[grp] = rsqrtf((float)(s * inv_hd) + eps);
    }
    __syncthreads();
  }
  const int half = HD / 2, npair = D / 2, rot_heads = H / 2;
  const float* fr = freqs + (size_t)p * HD;  // (位置, hd / 2, 2): cos, sin
  for (int j = threadIdx.x; j < 3 * npair; j += nt){
    int which = j / npair, rem = j % npair;
    int h = rem / half, i = rem % half;
    const wd_t* src = row + (size_t)which * D + h * HD + 2 * i;
    if (which == 2){
      size_t dst = ((size_t)(b * H + h) * K + p) * HD + 2 * i;
      v[dst] = src[0];
      v[dst + 1] = src[1];
      continue;
    }
    float r = rs[which * H + h];
    const float* w = qk_w + (size_t)which * D + h * HD + 2 * i;
    float a0 = LD(src[0]) * r * w[0], a1 = LD(src[1]) * r * w[1];
    if (h < rot_heads){
      float c = fr[2 * i], sn = fr[2 * i + 1];
      float t0 = a0 * c - a1 * sn, t1 = a0 * sn + a1 * c;
      a0 = t0; a1 = t1;
    }
    if (which == 0){
      size_t dst = ((size_t)(b * H + h) * L + p) * HD + 2 * i;
      q[dst] = ST(a0); q[dst + 1] = ST(a1);
    } else {
      size_t dst = ((size_t)(b * H + h) * K + p) * HD + 2 * i;
      k[dst] = ST(a0); k[dst + 1] = ST(a1);
    }
  }
}

// 1 ブロック = 行 (b, p)。out[b, p, h × hd + e] = y[b, h, p, e] × sigmoid(o4[b, p, 3D + h × hd + e])
extern "C" __global__ void gate(const wd_t* y, const wd_t* o4, ot_t* out, int L, int H, int HD, float os){
  const int D = H * HD;
  const int b = blockIdx.x / L, p = blockIdx.x % L;
  const wd_t* g = o4 + (size_t)blockIdx.x * 4 * D + 3 * D;
  ot_t* o = out + (size_t)blockIdx.x * D;
  for (int j = threadIdx.x; j < D; j += NT){
    int h = j / HD, e = j % HD;
    float yv = LD(y[((size_t)(b * H + h) * L + p) * HD + e]);
    o[j] = STO(yv * (1.0f / (1.0f + expf(-LD(g[j])))), os);
  }
}

// 1 ブロック = 1 行。out = silu(a) × b。o13 の行は [a（hidden 個）, b（hidden 個）]。
extern "C" __global__ void swiglu(const wd_t* o13, ot_t* out, int hidden, float os){
  const wd_t* row = o13 + (size_t)blockIdx.x * 2 * hidden;
  ot_t* o = out + (size_t)blockIdx.x * hidden;
  for (int j = threadIdx.x; j < hidden; j += NT){
    float a = LD(row[j]);
    o[j] = STO(a / (1.0f + expf(-a)) * LD(row[hidden + j]), os);
  }
}

// CFG の合成と x の更新。raw: (3 か 1, L, C)。x: (1, L, C)。v = c0 + s_text (c0 − c1) + s_spk (c0 − c2)、x += v × dt
extern "C" __global__ void cfg_update(const float* raw, float* x, const float* dt, int n, int cfg, float s_text, float s_spk){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float c0 = raw[i], vv = c0;
  if (cfg){
    vv = c0 + s_text * (c0 - raw[n + i]);
    vv = vv + s_spk * (c0 - raw[2 * n + i]);
  }
  x[i] = x[i] + vv * dt[0];
}
'''


def _ptr(t: torch.Tensor) -> np.uintp:
    """テンソルの先頭のアドレス。カーネルのポインターの引数に、64 ビットの整数として渡す。
    CuPy の配列に包むより、記録のときの CPU の時間が短い（起動の準備で 5 万回ほど呼ぶ）。"""
    assert t.is_contiguous(), "カーネルに渡すテンソルは連続でなければならない"
    return np.uintp(t.data_ptr())


class FusedCore(DitCore):
    """DitCore と同じ計算を、まとめたカーネルで行う。x は (B, L, 潜在の次元)、マスクは足し込む値で受ける。"""

    KERNELS = ("adaln", "res_adaln", "qkv", "gate", "swiglu", "cfg_update")
    EXTRA_OPTS: tuple[str, ...] = ()

    def __init__(self, model, wdtype: torch.dtype):
        super().__init__(model, wdtype)
        import cupy as cp

        self.cp = cp
        t0 = time.perf_counter()
        # qkv の 1 ブロックのスレッドの数（2 × heads × 8）。共有メモリーの配列の大きさも、これで決める（Small は 320）。
        self.qkv_threads = 2 * self.heads * 8
        opts = ("-std=c++14", f"-DQKV_NT={self.qkv_threads}") + (("-DWD_BF16",) if wdtype == torch.bfloat16 else ()) + self.EXTRA_OPTS
        mod = cp.RawModule(code=_SRC, options=opts)
        self.k = {n: mod.get_function(n) for n in self.KERNELS}
        self.compile_sec = time.perf_counter() - t0
        self.act_dtype = wdtype  # 行列積に入れる値の型
        self.tap = None  # 関数を入れると、行列積に入れる値を作るたびに (層, 場所, テンソル) で呼ぶ（グラフなしのときだけ使う）
        assert wdtype in (torch.bfloat16, torch.float32)
        assert self.heads % 2 == 0 and self.head_dim % 2 == 0 and self.qkv_threads <= 1024 and self.head_dim % 8 == 0
        for p in self.layers:
            p["qk_w"] = p["qk_weight"].reshape(-1).contiguous()
            p["a_eps_f"], p["m_eps_f"], p["qk_eps_f"] = (float(p[n].item()) for n in ("a_eps", "m_eps", "qk_eps"))

    def stream(self):
        """PyTorch の今のストリームを CuPy の今のストリームにする。カーネルはこの中で起動する
        （CUDA Graph の記録は、PyTorch が記録中のストリームへの起動を拾う）。"""
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)

    def launch(self, name: str, blocks: int, threads: int, *args) -> None:
        """stream() の中で呼ぶ。"""
        self.k[name]((int(blocks),), (int(threads),), tuple(_ptr(a) if isinstance(a, torch.Tensor) else a for a in args))

    def act_scale(self, li: int, site: str) -> float:
        """行列積に入れる値に掛ける倍率。場所は ha（attention の前）、yg（wo の前）、hm（MLP の前）、ym（w2 の前）。"""
        return 1.0

    def mm(self, x: torch.Tensor, li: int, name: str, site: str) -> torch.Tensor:
        """層 li の重み name との行列積。x は site の値。"""
        return F.linear(x, self.layers[li][name])

    def __call__(self, x, coef, ctx_k, ctx_v, bias, freqs_real, return_blocks: bool = False):
        """x: (B, L, 32)。coef: (層, 6, 1, 1, D)。ctx_k・ctx_v: (層, 1, S, heads, head_dim)。
        bias: (B, 1, 1, L + S)、行列積の型で 0 か −inf。freqs_real: (位置, head_dim / 2, 2) の fp32。"""
        bsz, seq_len = x.shape[0], x.shape[1]
        heads, hd, dim, wd = self.heads, self.head_dim, self.dim, self.wdtype
        n_ctx = int(ctx_k.shape[2])
        rows, keys = bsz * seq_len, seq_len + n_ctx
        i32, f32 = np.int32, np.float32
        dev = x.device
        h_res = self.in_proj(x).reshape(rows, dim).contiguous()
        adt, S, tap = self.act_dtype, self.act_scale, self.tap
        h = torch.empty((rows, dim), device=dev, dtype=adt)
        q = torch.empty((bsz, heads, seq_len, hd), device=dev, dtype=wd)
        k = torch.empty((bsz, heads, keys, hd), device=dev, dtype=wd)
        v = torch.empty((bsz, heads, keys, hd), device=dev, dtype=wd)
        yg = torch.empty((rows, dim), device=dev, dtype=adt)
        ym = torch.empty((rows, self.layers[0]["hidden"]), device=dev, dtype=adt)
        with self.stream():
            outs = []
            c = coef[0]
            self.launch("adaln", rows, 256, h_res, c[0], c[1], h, i32(dim), f32(self.inv_dim), f32(self.layers[0]["a_eps_f"]),
                        f32(S(0, "ha")))
            for li, p in enumerate(self.layers):
                c = coef[li]
                if tap is not None:
                    tap(li, "ha", h)
                o4 = self.mm(h, li, "w_qkvg", "ha")  # (行, 4D): q、k、v、gate
                self.launch("qkv", bsz * keys, self.qkv_threads, o4, p["qk_w"], freqs_real, ctx_k[li], ctx_v[li], q, k, v,
                            i32(seq_len), i32(n_ctx), i32(heads), i32(hd), f32(self.inv_head_dim), f32(p["qk_eps_f"]))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                self.launch("gate", rows, 256, y.contiguous(), o4, yg, i32(seq_len), i32(heads), i32(hd), f32(S(li, "yg")))
                if tap is not None:
                    tap(li, "yg", yg)
                o = self.mm(yg, li, "w_o", "yg")
                self.launch("res_adaln", rows, 256, h_res, o, c[2], c[3], c[4], h, i32(dim), f32(self.inv_dim), f32(p["m_eps_f"]),
                            f32(S(li, "hm")))
                if tap is not None:
                    tap(li, "hm", h)
                o13 = self.mm(h, li, "w_13", "hm")
                self.launch("swiglu", rows, 256, o13, ym, i32(p["hidden"]), f32(S(li, "ym")))
                if tap is not None:
                    tap(li, "ym", ym)
                o = self.mm(ym, li, "w_2", "ym")
                if li + 1 < len(self.layers):
                    cn = coef[li + 1]
                    self.launch("res_adaln", rows, 256, h_res, o, c[5], cn[0], cn[1], h, i32(dim), f32(self.inv_dim),
                                f32(self.layers[li + 1]["a_eps_f"]), f32(S(li + 1, "ha")))
                    if return_blocks:
                        outs.append(h_res.reshape(bsz, seq_len, dim).clone())
                else:
                    h_res = torch.addcmul(h_res, c[5].reshape(1, dim), o)
                    if return_blocks:
                        outs.append(h_res.reshape(bsz, seq_len, dim))
        out = self.out_proj(self.out_norm(h_res.reshape(bsz, seq_len, dim)))
        if return_blocks:
            return out, outs
        return out


class FusedBucketSampler(BucketSampler):
    """BucketSampler の 1 ステップを FusedCore で計算する。バケット、グラフの持ち方、呼び出し口は同じ。"""

    def __init__(self, model, *, precision: str = "bf16", fp8: bool | str | None = None, fp8_headroom: float = 4.0, **kw):
        """fp8: DiT の行列積を FP8（e4m3）にするステップ。None（しない。既定）、"all"（True も同じ）、"cfg"、"nocfg"。
        出力が変わるので、既定では使わない（engine/fp8.py）。"""
        super().__init__(model, precision=precision, **kw)
        self.fp8 = {True: "all", False: None}.get(fp8, fp8)
        assert self.fp8 in (None, "all", "cfg", "nocfg"), fp8
        assert self.fp8 is None or precision == "bf16", "FP8 は BF16 のエンジンに重ねて使う"
        assert self.fp8 is None or not self.meanflow, "FP8 は RF 版でだけ確かめた（MF 版では使わない）"
        self.fcore8 = None
        self._calibrating = False
        with torch.inference_mode():
            self.fcore = FusedCore(model, self.wdtype)
            if self.fp8:
                from engine.fp8 import Fp8Core

                self.fcore8 = Fp8Core(model, headroom=fp8_headroom)
            # RoPE の表を実数の組 (位置, head_dim / 2, 2) にしたもの。
            self.freqs_real = torch.view_as_real(self.freqs[0, :, 0, 0, :]).contiguous()
        self.kernel_compile_sec = self.fcore.compile_sec
        self._bias_bufs: dict[tuple[int, int, int], torch.Tensor] = {}

    def _shape(self, lb: int, tb: int, n_spk: int) -> _Shape:
        key = (lb, tb, n_spk)
        if key not in self.shapes:  # グラフの外に置くバッファなので、記録の前に確保する
            self._bias_bufs[key] = torch.zeros((3, 1, 1, lb + tb + n_spk), device=self.device, dtype=self.wdtype)
            self._recording = key
        try:
            return super()._shape(lb, tb, n_spk)
        finally:
            self._recording = None

    def _bias(self, sh: _Shape) -> torch.Tensor:
        """attention に足し込むマスク（0 か −inf）。形ごとに 1 つ持つ。
        記録する形は、記録の前に確保したバッファを使う。確かめ用の形（tf_prepare）は、自分のバッファを持つ。"""
        b = getattr(sh, "bias3", None)
        if b is None:
            key = (sh.lb, sh.tb, sh.n_spk)
            b = self._bias_bufs[key] if getattr(self, "_recording", None) == key else \
                torch.zeros(sh.mask3.shape, device=self.device, dtype=self.wdtype)
            sh.bias3 = b
            sh.bias_for = None
        return b

    def _fill(self, sh: _Shape, seq_len: int, n_text: int, voice: Voice, tk: torch.Tensor, tv: torch.Tensor) -> None:
        super()._fill(sh, seq_len, n_text, voice, tk, tv)
        b = self._bias(sh)
        if sh.bias_for != sh.filled:
            b.zero_().masked_fill_(~sh.mask3, float("-inf"))
            sh.bias_for = sh.filled

    # ---- FP8（選べる設定。engine/fp8.py） ----

    def _core(self, cfg: bool) -> FusedCore:
        """このステップを計算する本体。FP8 にするステップでは Fp8Core。"""
        if self.fp8 is None or self._calibrating or (self.fp8 == "cfg" and not cfg) or (self.fp8 == "nocfg" and cfg):
            return self.fcore
        return self.fcore8

    def _ensure_fp8(self, voice: Voice) -> None:
        """FP8 の倍率を、最初の声で一度だけ決める。グラフを記録する前に呼ぶ（倍率はグラフに焼き込まれる）。"""
        if self.fcore8 is None or self.fcore8.calibrated or self._calibrating:
            return
        from engine.fp8 import calibrate

        self._calibrating = True
        try:
            self.fp8_calibration = calibrate(self, voice)
        finally:
            self._calibrating = False

    def prewarm(self, voice: Voice) -> dict:
        with torch.inference_mode():
            self._ensure_fp8(voice)
        info = super().prewarm(voice)
        if self.fcore8 is not None:
            info["fp8"] = self.fp8
            info["fp8_calibrate_sec"] = self.fp8_calibration["sec"]
        return info

    def tf_prepare(self, seq_len: int, text_state: torch.Tensor, voice: Voice) -> _Shape:
        self._ensure_fp8(voice)
        return super().tf_prepare(seq_len, text_state, voice)

    def sample(self, **kw) -> torch.Tensor:
        if self.fcore8 is not None and not self.fcore8.calibrated and kw.get("model") is self.model and self._supported(kw):
            self._ensure_fp8(self.voice_for(kw["ref_latent"], kw["ref_mask"]))
        return super().sample(**kw)

    def _body(self, sh: _Shape, cfg: bool):
        """1 ステップ。sh.x をその場で更新し、ステップの番号を 1 進める。戻り値は (CFG を合成した後の v, モデルの出力)。"""
        coef = self.coef.index_select(0, self.idx)[0].contiguous()
        dt = self.dt_t.index_select(0, self.idx)
        x = sh.x
        bias3 = self._bias(sh)
        fcore = self._core(cfg)
        if cfg:
            raw = fcore(x.expand(3, -1, -1), coef, sh.ctx.k, sh.ctx.v, bias3, self.freqs_real)
            c0, c1, c2 = raw[0:1], raw[1:2], raw[2:3]
            v = c0 + self.scales[0] * (c0 - c1)
            v = v + self.scales[1] * (c0 - c2)
        else:
            raw = fcore(x, coef, sh.ctx.k, sh.ctx.v, bias3[0:1], self.freqs_real)
            v = raw
        x.add_(v * dt)
        self.idx.add_(1)
        return v, raw

    def _body_fast(self, sh: _Shape, cfg: bool) -> None:
        """_body と同じ 1 ステップ。CFG の合成と x の更新を 1 個のカーネルで行う（グラフに記録するのはこちら）。"""
        coef = self.coef.index_select(0, self.idx)[0].contiguous()
        dt = self.dt_t.index_select(0, self.idx)
        bias3 = self._bias(sh)
        x = sh.x
        fcore = self._core(cfg)
        if cfg:
            raw = fcore(x.expand(3, -1, -1), coef, sh.ctx.k, sh.ctx.v, bias3, self.freqs_real)
        else:
            raw = fcore(x, coef, sh.ctx.k, sh.ctx.v, bias3[0:1], self.freqs_real)
        n = x.numel()
        with self.fcore.stream():
            self.fcore.launch("cfg_update", -(-n // 256), 256, raw.contiguous(), x, dt, np.int32(n), np.int32(1 if cfg else 0),
                              np.float32(self.scales[0]), np.float32(self.scales[1]))
        self.idx.add_(1)

    def _steps(self, sh: _Shape, cfg: bool) -> None:
        with matmul_tf32(self.precision == "tf32"):
            for _ in range(self.k):
                self._body_fast(sh, cfg)


def install_fused(rt, *, precision: str = "bf16", **kw) -> FusedBucketSampler:
    """公式の runtime にエンジンを差し込む。復号・電子透かし・後処理は公式のまま。"""
    sampler = FusedBucketSampler(rt.model, precision=precision, **kw)
    patch_runtime(sampler)
    return sampler


def half_decode(rt, dtype: torch.dtype = torch.float16) -> None:
    """復号（DACVAE）を半精度の autocast で動かす。出力が変わるので、既定では使わない。

    fp32 の復号との差は、90 フレームの乱数の潜在変数で SNR 59.5 dB（float16）、41.2 dB（bfloat16）だった
    （runs/experiments/decode-probe-default.txt）。時間は 13.9 → 9.3 ミリ秒。"""
    orig = rt.codec.decode_latent

    def decode_latent(latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return orig(latent).float()

    rt.codec.decode_latent = decode_latent
