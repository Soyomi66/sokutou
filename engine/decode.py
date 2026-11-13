"""復号（DACVAE）を、決まった長さ（バケット）で動かす。

公式の復号は、潜在変数の長さごとに形の違う畳み込みを呼ぶ。cuDNN は初めての形で畳み込みの準備をするので、
初めての長さの復号は 8〜14 ミリ秒遅い（CPU 側の `cudnn_convolution` の時間。runs/experiments/probe2.txt）。
BucketDecoder は、潜在変数を区切りの長さまで 0 で伸ばして復号し、起動のときにすべての区切りを一度通しておく。

伸ばした分が有効な部分の計算に混ざらないよう、復号器のすべての畳み込みの出力で、有効な長さより後ろを 0 にする。
復号器の畳み込みは端を 0 で埋めるので（NormConv1d の pad_mode "none" と、長さの変わらない "auto"）、
有効な部分から見ると、後ろが 0 で埋まっているのは、詰め物のない復号と同じ入力になる。活性化（Snake、Tanh）は
0 を 0 に写し、残差の足し算も 0 を保つ。転置畳み込みは、長さ L の入力からちょうど L × stride の出力を作る。
数学的には同じ計算だが、cuDNN が形ごとに選ぶ計算の方法が変わるので、値は丸めの範囲で変わる。

FusedBucketDecoder は、BucketDecoder と同じ値を少ないカーネルで計算する（畳み込みのあいだの bias・残差・Snake・
後ろを 0 にする処理を、自作のカーネル 1 個にまとめる）。cupy があれば、install_bucket_decoder はこちらを使う。
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def default_decode_buckets(max_len: int = 750) -> list[int]:
    """復号の長さの区切り（フレーム）。256 まで 8 刻み、その先は 32 刻み（30 秒 = 750 フレームまでを覆う）。
    起動の準備の時間は長い区切りで決まるので、長いほうを粗くする。"""
    out = list(range(16, 257, 8)) + list(range(288, max_len + 32, 32))
    return [b for b in out if b < max_len + 32]


class BucketDecoder:
    def __init__(self, rt, buckets: list[int] | None = None):
        self.rt = rt
        self.codec = rt.codec
        self.buckets = sorted(buckets or default_decode_buckets())
        self.orig = rt.codec.decode_latent
        self.hop = int(rt.codec.model.hop_length)
        self.state: tuple[int, int] | None = None  # (有効なフレーム, 区切りのフレーム)。None なら何もしない
        self.warmed: set[int] = set()
        self.misses = 0  # 区切りを超える長さ（公式の復号をそのまま呼んだ回数）
        self.cold = 0  # 起動のときに通していない区切りを使った回数
        model = rt.codec.model
        convs = [m for m in model.decoder.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))]
        convs.append(model.quantizer.out_proj)
        self.handles = [m.register_forward_hook(self._zero_tail) for m in convs]

    def _zero_tail(self, module, inputs, out):
        st = self.state
        if st is not None:
            total = out.shape[-1]
            assert (total * st[0]) % st[1] == 0, "出力の長さが入力の長さの整数倍でない"
            valid = total * st[0] // st[1]
            if valid < total:
                out[..., valid:] = 0
        return None

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """公式の DACVAECodec.decode_latent と同じ引数と戻り値。"""
        frames = int(latent.shape[1])
        bucket = next((b for b in self.buckets if frames <= b), None)
        if bucket is None:
            self.misses += 1
            return self.orig(latent)
        if bucket not in self.warmed:
            self.cold += 1
            self.warmed.add(bucket)
        if bucket == frames:
            return self.orig(latent)
        z = latent.new_zeros((latent.shape[0], bucket, latent.shape[2]))
        z[:, :frames] = latent
        self.state = (frames, bucket)
        try:
            audio = self.orig(z)
        finally:
            self.state = None
        return audio[..., : frames * self.hop]

    @torch.inference_mode()
    def prewarm(self) -> dict:
        """すべての区切りを一度通す（長い順。PyTorch が確保したメモリーを、短い区切りが使い回しやすい）。"""
        t0 = time.perf_counter()
        dim = int(self.rt.model_cfg.latent_dim)
        g = torch.Generator().manual_seed(0)
        for b in sorted(self.buckets, reverse=True):
            z = torch.randn(1, b, dim, generator=g).to(self.codec.device)
            self.decode_latent(z[:, : b - 1]).cpu()  # 詰め物ありの経路（後ろを 0 にする処理を含む）
            self.warmed.add(b)
        torch.cuda.synchronize()
        self.cold = 0
        return {"prewarm_decode_sec": time.perf_counter() - t0, "decode_buckets": len(self.buckets)}


_FUSE_SRC = r'''
// 畳み込み（bias なし）の出力 c に、bias と残差を足し、Snake を掛ける。1 スレッド = 1 要素。並びは (C, T)。
// ブロックの y がチャンネル、x × スレッドが時刻（要素ごとの割り算をしない）。
//   v = c + bias[ch]（x があれば、さらに + x）。t >= valid なら v = 0（詰め物の位置）
//   out_x があれば v を書く。out_a があれば snake(v) = v + (1 / (alpha + 1e-9)) × sin(alpha × v)² を書く。
// Snake の式は、公式（TorchScript がまとめたカーネル）とビット単位で同じ結果になる書き方にしてある
// （float の 1e-9、s × s、最後の掛け算と足し算は fmaf。runs/experiments/scripts/fable_snake_probe.py で確かめた）。
// out_x と out_a は c と同じ場所でもよい（要素ごとに読んでから書く）。
extern "C" __global__ void fuse(const float* c, const float* bias, const float* x, const float* alpha,
                                float* out_x, float* out_a, int T, int valid){
  const int ch = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const size_t i = (size_t)ch * T + t;
  float v = 0.0f;
  if (t < valid){
    v = __fadd_rn(c[i], bias[ch]);
    if (x) v = __fadd_rn(v, x[i]);
  }
  if (out_x) out_x[i] = v;
  if (out_a){
    float a = alpha[ch];
    float s = sinf(a * v);
    out_a[i] = fmaf(1.0f / (a + 1e-9f), s * s, v);
  }
}
'''


class FusedBucketDecoder(BucketDecoder):
    """BucketDecoder と同じ値を、少ないカーネルで計算する。

    公式の復号器は、畳み込みのあとに bias の足し算、残差の足し算、Snake を別々のカーネルで計算し、BucketDecoder は
    さらに畳み込みごとに後ろを 0 にする。ここでは畳み込み（cuDNN。公式と同じ呼び方で、bias だけ外す）のあいだの処理を、
    自作のカーネル 1 個（fuse）にまとめる。weight_norm の重みは、公式と同じ関数で一度だけ計算しておく。
    足し算の順序と Snake の式は公式と同じなので、出力は BucketDecoder とビット単位で同じになる（確かめた範囲で）。
    cupy が要る（engine/fused.py と同じ）。バッチは 1、復号器は fp32 だけを扱い、ほかは BucketDecoder の経路に戻る。
    """

    def __init__(self, rt, buckets: list[int] | None = None):
        super().__init__(rt, buckets)
        import cupy as cp
        from dacvae.nn.layers import Snake1d

        self.cp = cp
        self.kernel = cp.RawModule(code=_FUSE_SRC, options=("-std=c++14",)).get_function("fuse")
        model = rt.codec.model
        dec = model.decoder
        self.out_proj = model.quantizer.out_proj
        self.first = self._conv(dec.model[0])
        self.blocks = []
        for blk in dec.model[1:]:
            layers = list(blk.block)
            assert isinstance(layers[0], Snake1d) and isinstance(layers[1], torch.nn.ConvTranspose1d)
            units = []
            for unit in (layers[4], layers[5], layers[8]):  # DecoderBlock.forward が通す 3 つの ResidualUnit
                a1, c1, a2, c2 = unit.block
                assert isinstance(a1, Snake1d) and isinstance(a2, Snake1d) and not unit.true_skip
                assert c1.padding[0] * 2 == (c1.kernel_size[0] - 1) * c1.dilation[0] and c2.kernel_size[0] == 1
                units.append((self._alpha(a1), self._conv(c1), self._alpha(a2), self._conv(c2)))
            assert isinstance(layers[9], torch.nn.Identity), "最後の ResidualUnit がある復号器は扱っていない"
            self.blocks.append((self._alpha(layers[0]), self._conv(layers[1]), units))
        pre = dec.wm_model.encoder_block.pre  # 公式の _watermark_passthrough: Snake → 畳み込み → Tanh
        assert isinstance(pre[0], Snake1d) and isinstance(pre[2], torch.nn.Tanh) and len(pre) == 4
        self.last_alpha, self.last_conv = self._alpha(pre[0]), pre[1]
        self.fused_ok = rt.codec.dtype == torch.float32
        self.fused_calls = 0

    @staticmethod
    def _alpha(snake) -> torch.Tensor:
        return snake.alpha.detach().reshape(-1).contiguous()

    @staticmethod
    def _conv(m) -> dict:
        """畳み込みの重み（weight_norm は公式と同じ関数で一度だけ計算する）と bias と引数。"""
        w = m.weight
        for hook in m._forward_pre_hooks.values():
            if type(hook).__name__ == "WeightNorm":
                w = hook.compute_weight(m)
        assert m.groups == 1 and m.bias is not None and getattr(m, "pad_mode", "none") == "none"
        tr = isinstance(m, torch.nn.ConvTranspose1d)
        return {"w": w.detach().contiguous(), "b": m.bias.detach().contiguous(), "tr": tr, "stride": m.stride, "padding": m.padding,
                "dilation": m.dilation, "output_padding": m.output_padding if tr else None}

    @staticmethod
    def _run(cv: dict, x: torch.Tensor) -> torch.Tensor:
        if cv["tr"]:
            return F.conv_transpose1d(x, cv["w"], None, cv["stride"], cv["padding"], cv["output_padding"], 1, cv["dilation"])
        return F.conv1d(x, cv["w"], None, cv["stride"], cv["padding"], cv["dilation"], 1)

    def _fuse(self, c: torch.Tensor, bias: torch.Tensor, x: torch.Tensor | None, alpha: torch.Tensor | None,
              valid: int, keep_x: bool) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """戻り値は (v, snake(v))。keep_x なら v を c の場所に書き、snake(v) は新しい場所に書く。そうでなければ snake(v) を c の場所に書く。"""
        assert c.is_contiguous() and c.shape[0] == 1 and (x is None or x.shape == c.shape)
        C, T = int(c.shape[1]), int(c.shape[2])
        out_a = None if alpha is None else (torch.empty_like(c) if keep_x else c)
        ptr = lambda t: np.uintp(0 if t is None else t.data_ptr())  # noqa: E731
        self.kernel((-(-T // 256), C), (256,), (ptr(c), ptr(bias), ptr(x), ptr(alpha), ptr(c if keep_x else None), ptr(out_a),
                                             np.int32(T), np.int32(valid)))
        return (c if keep_x else None), out_a

    def _forward(self, z: torch.Tensor, frames: int) -> torch.Tensor:
        """z: (1, D, 区切りの長さ)。frames より後ろは詰め物。戻り値は公式の model.decode と同じ (1, 1, サンプル)。"""
        with self.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            valid = frames
            h = self.out_proj(z)
            if valid < h.shape[-1]:
                h[..., valid:] = 0
            c, bias = self._run(self.first, h), self.first["b"]
            x = None  # 残差に足す値（ブロックの最初の Snake の前では足さない）
            for alpha0, convt, units in self.blocks:
                _, a = self._fuse(c, bias, x, alpha0, valid, False)
                valid *= convt["stride"][0]
                c, bias = self._run(convt, a), convt["b"]
                x = None
                for a1, c1, a2, c2 in units:
                    x, a = self._fuse(c, bias, x, a1, valid, True)
                    _, a = self._fuse(self._run(c1, a), c1["b"], None, a2, valid, False)
                    c, bias = self._run(c2, a), c2["b"]
            _, a = self._fuse(c, bias, x, self.last_alpha, valid, False)
            return torch.tanh(self.last_conv(a))

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """公式の DACVAECodec.decode_latent と同じ引数と戻り値。"""
        frames = int(latent.shape[1])
        bucket = next((b for b in self.buckets if frames <= b), None)
        if bucket is None or not self.fused_ok or latent.ndim != 3 or latent.shape[0] != 1:
            return super().decode_latent(latent)
        if bucket not in self.warmed:
            self.cold += 1
            self.warmed.add(bucket)
        self.fused_calls += 1
        z = torch.zeros((1, int(latent.shape[2]), bucket), device=self.codec.device, dtype=self.codec.dtype)
        z[:, :, :frames] = latent.transpose(1, 2)
        return self._forward(z, frames)[..., : frames * self.hop]


def make_bucket_decoder(rt, buckets: list[int] | None = None, fused: bool | None = None) -> BucketDecoder:
    """区切りの長さで動かす復号を作る（runtime には差し込まない）。

    fused: 自作のカーネルにまとめた FusedBucketDecoder を使うか。None なら、cupy があれば使う
    （環境変数 SOKUTOU_FUSED_DECODE=0 なら使わない）。出力は BucketDecoder と同じ。"""
    if fused is None:
        # half_decode などで復号が差し替えてあるときは、その関数を通す BucketDecoder にする
        official = getattr(rt.codec.decode_latent, "__self__", None) is rt.codec
        fused = os.environ.get("SOKUTOU_FUSED_DECODE", "1") != "0" and importlib.util.find_spec("cupy") is not None \
            and rt.codec.dtype == torch.float32 and official
    return FusedBucketDecoder(rt, buckets) if fused else BucketDecoder(rt, buckets)


def install_bucket_decoder(rt, buckets: list[int] | None = None, fused: bool | None = None) -> BucketDecoder:
    """runtime の復号を、区切りの長さで動かすものに差し替える。half_decode と併用するときは、half_decode を先に呼ぶ。
    fused は make_bucket_decoder と同じ。"""
    dec = make_bucket_decoder(rt, buckets, fused)
    rt.codec.decode_latent = dec.decode_latent
    return dec
