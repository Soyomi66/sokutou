"""DiT の 12 層の行列積を FP8（e4m3）で計算する本体（選べる設定。出力が変わるので、既定では使わない）。

FusedCore（engine/fused.py、BF16）との違いは、1 層に 4 つある行列積（wq・wk・wv・gate をつないだもの、wo、
w1・w3 をつないだもの、w2）だけ。
- 重みは、テンソルごとの倍率（2 の累乗）を掛けて e4m3 に丸めた複製を持つ。
- 行列積に入れる値（AdaLN の出力、ゲートを掛けた attention の出力、SwiGLU の出力）は、それを作るカーネルが
  fp32 で計算した値に、場所ごとの決まった倍率（2 の累乗）を掛けて、e4m3 で直接書く。448 を超える値は ±448 にする。
- 行列積は `torch._scaled_mm`（cuBLASLt。sm_89 の FP8 のテンソルコアを使う）。足し込みは fp32、出力は BF16。
  倍率の逆数を行列積の中で掛けるので、出力は BF16 のエンジンと同じ大きさになる。
- 残差の流れ、AdaLN、q と k の正規化と RoPE、attention、sigmoid・silu、in_proj、out_norm、out_proj、CFG の合成、
  x の更新、条件の K・V の射影、文と話者のエンコーダー、復号は、BF16 のエンジンと同じ（fp32 か BF16）。

e4m3 は、符号 1 ビット、指数 4 ビット、仮数 3 ビットの浮動小数点数（最大 448、正規の数の最小 2^-6、その下は
2^-9 刻み）。仮数が 3 ビットなので、1 つの値の丸めの誤差は相対で約 2.7%（重みの SNR で約 31.5 dB）。浮動小数点数
なので、倍率は誤差の大きさをほとんど変えない。倍率は、値が 448 を超えないことと、小さな値が 2^-6 を下回って粗く
ならないことのために使う。

倍率は、グラフに焼き込むので、合成の途中で変えない。最初の声で一度だけ、BF16 の本体を通して決める（calibrate）。
グラフの中で最大値から求める方法は採らない。詰め物の行や CFG のほかの行の値で、有効な行の結果が変わるため。

`torch._scaled_mm` は PyTorch の非公開の API。重みの列数と行数は 16 の倍数でなければならない（このモデルは
1280・5120・7360・3680 で、満たす）。行数（潜在の長さ）に制約はない（runs/experiments/fp8-probe.txt）。
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from engine.fused import FusedCore

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
SITES = ("ha", "yg", "hm", "ym")  # 行列積に入れる値の場所: attention の前、wo の前、MLP の前、w2 の前
WEIGHT_SITE = {"w_qkvg": "ha", "w_o": "yg", "w_13": "hm", "w_2": "ym"}


def pow2_scale(max_abs: float, headroom: float = 1.0) -> float:
    """max_abs × headroom × 倍率 が 448 以下になる、いちばん大きい 2 の累乗。"""
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0
    return float(2.0 ** math.floor(math.log2(E4M3_MAX / (max_abs * headroom))))


class Fp8Core(FusedCore):
    KERNELS = FusedCore.KERNELS + ("e4m3_encode",)
    EXTRA_OPTS = ("-DOUT_E4M3",)

    def __init__(self, model, headroom: float = 4.0):
        super().__init__(model, torch.bfloat16)
        assert hasattr(torch, "_scaled_mm"), "この PyTorch には torch._scaled_mm がない"
        self.act_dtype = E4M3
        self.headroom = float(headroom)
        self.calibrated = False
        dev = model.device
        self.w8: list[dict[str, torch.Tensor]] = []  # 層ごとの、e4m3 の重み（転置して渡す）と倍率の逆数
        self.weight_scale: list[dict[str, float]] = []
        for p in self.layers:
            d, sc = {}, {}
            for name in WEIGHT_SITE:
                w = p[name].float()
                assert w.shape[0] % 16 == 0 and w.shape[1] % 16 == 0, "torch._scaled_mm は 16 の倍数の形だけを扱う"
                s = pow2_scale(float(w.abs().max()))
                d[name] = (w * s).to(E4M3).contiguous().t()  # (入力, 出力) の、列の向きに並んだ行列
                d[name + "_inv"] = torch.tensor(1.0 / s, device=dev, dtype=torch.float32)
                sc[name] = s
            self.w8.append(d)
            self.weight_scale.append(sc)
        self.scale = {(li, site): 1.0 for li in range(len(self.layers)) for site in SITES}
        self.inv = {k: torch.tensor(1.0, device=dev, dtype=torch.float32) for k in self.scale}

    def set_scales(self, max_abs: dict[tuple[int, str], float]) -> None:
        """場所ごとの最大の絶対値から、倍率を決める。グラフを記録する前に一度だけ呼ぶ。"""
        assert not self.calibrated, "倍率はグラフに焼き込まれるので、決め直さない"
        for key, m in max_abs.items():
            s = pow2_scale(m, self.headroom)
            self.scale[key] = s
            self.inv[key].fill_(1.0 / s)
        self.calibrated = True

    def act_scale(self, li: int, site: str) -> float:
        return self.scale[(li, site)]

    def mm(self, x: torch.Tensor, li: int, name: str, site: str) -> torch.Tensor:
        d = self.w8[li]
        return torch._scaled_mm(x, d[name], self.inv[(li, site)], d[name + "_inv"], out_dtype=torch.bfloat16)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 のテンソルを、カーネルの中と同じ関数で e4m3 にする（確かめ用）。"""
        x = x.float().contiguous()
        out = torch.empty(x.shape, device=x.device, dtype=E4M3)
        with self.stream():
            self.launch("e4m3_encode", -(-x.numel() // 256), 256, x, out, np.int32(x.numel()))
        return out


@torch.inference_mode()
def calibrate(sampler, voice, *, seq_len: int = 100, n_text: int = 20, seed: int = 0) -> dict:
    """行列積に入れる値の最大の絶対値を、BF16 の本体で 1 文ぶん（全ステップ、CFG あり・なし）通して測り、
    Fp8Core の倍率を決める。入力は作った値（決まった乱数の文のトークンとノイズ）と、この声の話者の条件。
    実際の文との違いは、倍率の余裕（headroom、既定 4 倍）と、448 での打ち切りで受ける。
    実際の文で測った最大値との比は runs/experiments/fp8-calib.txt にある。"""
    t0 = time.perf_counter()
    stats = measure_max(sampler, voice, None, seq_len=seq_len, n_text=n_text, seed=seed)
    sampler.fcore8.set_scales(stats)
    return {"sec": time.perf_counter() - t0, "max_abs": {f"{li}.{site}": v for (li, site), v in stats.items()},
            "scale": {f"{li}.{site}": v for (li, site), v in sampler.fcore8.scale.items()}}


@torch.inference_mode()
def measure_max(sampler, voice, text_ids: torch.Tensor | None, *, seq_len: int = 100, n_text: int = 20, seed: int = 0,
                x0: torch.Tensor | None = None) -> dict[tuple[int, str], float]:
    """BF16 の本体（グラフなし）で全ステップを通し、場所ごとの最大の絶対値を返す。text_ids が None なら乱数のトークン。"""
    dev = sampler.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    if text_ids is None:
        text_ids = torch.randint(5, 1000, (1, n_text), generator=g).to(dev)
    if x0 is None:
        x0 = torch.randn((1, seq_len, sampler.model.cfg.patched_latent_dim), generator=g).to(dev)
    stats: dict[tuple[int, str], torch.Tensor] = {}

    def tap(li: int, site: str, t: torch.Tensor) -> None:
        m = t.float().abs().max()
        key = (li, site)
        stats[key] = m if key not in stats else torch.maximum(stats[key], m)

    was, sampler._calibrating = sampler._calibrating, True  # BF16 の本体で通す
    sampler.fcore.tap = tap
    try:
        sh = sampler.tf_prepare(int(x0.shape[1]), sampler.encode_text(text_ids), voice)
        sampler._load_x(sh, x0)
        sampler.idx.zero_()
        for i in range(sampler.num_steps):
            sampler._body(sh, sampler.use_cfg[i])
    finally:
        sampler.fcore.tap = None
        sampler._calibrating = was
        sampler.idx.zero_()
    return {k: float(v) for k, v in stats.items()}
