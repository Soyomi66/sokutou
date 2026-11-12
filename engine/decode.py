"""復号（DACVAE）を、決まった長さ（バケット）で動かす。

公式の復号は、潜在変数の長さごとに形の違う畳み込みを呼ぶ。cuDNN は初めての形で畳み込みの準備をするので、
初めての長さの復号は 8〜14 ミリ秒遅い（CPU 側の `cudnn_convolution` の時間。runs/experiments/probe2.txt）。
BucketDecoder は、潜在変数を区切りの長さまで 0 で伸ばして復号し、起動のときにすべての区切りを一度通しておく。

伸ばした分が有効な部分の計算に混ざらないよう、復号器のすべての畳み込みの出力で、有効な長さより後ろを 0 にする。
復号器の畳み込みは端を 0 で埋めるので（NormConv1d の pad_mode "none" と、長さの変わらない "auto"）、
有効な部分から見ると、後ろが 0 で埋まっているのは、詰め物のない復号と同じ入力になる。活性化（Snake、Tanh）は
0 を 0 に写し、残差の足し算も 0 を保つ。転置畳み込みは、長さ L の入力からちょうど L × stride の出力を作る。
数学的には同じ計算だが、cuDNN が形ごとに選ぶ計算の方法が変わるので、値は丸めの範囲で変わる。
"""

from __future__ import annotations

import time

import torch


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


def install_bucket_decoder(rt, buckets: list[int] | None = None) -> BucketDecoder:
    """runtime の復号を、区切りの長さで動かすものに差し替える。half_decode と併用するときは、half_decode を先に呼ぶ。"""
    dec = BucketDecoder(rt, buckets)
    rt.codec.decode_latent = dec.decode_latent
    return dec
