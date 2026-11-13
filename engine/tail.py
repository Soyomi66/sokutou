"""文末を探す find_flattening_point を、公式と同じ番号を返したまま速くする。

公式の関数（irodori_tts/inference_runtime.py）は、潜在変数（GPU の上）の先頭から 1 フレームずつ、後ろ 20 フレームの
窓の標準偏差と平均を計算して Python の if で判定する。1 フレームごとに GPU の結果を待つので、
長さに比例して時間がかかる（90 フレームで約 3 ミリ秒、750 フレームで約 25 ミリ秒）。

ここでは、潜在変数を一度だけ CPU に写し、すべての窓の平均と分散を float64 の累積和でまとめて計算する。
しきい値の少し外まで（余裕 MARGIN）を候補にし、候補だけを公式と同じ式（同じデバイス、同じ型）で前から順に判定して、
最初に通った番号を返す。判定そのものは公式の式なので、返す番号は公式と同じになる
（float64 の累積和の誤差が余裕より小さいかぎり。値が有限でないときは公式の関数をそのまま呼ぶ）。
"""

from __future__ import annotations

import numpy as np
import torch

MARGIN = 1e-3  # 候補を広げる幅（しきい値に対する割合と、絶対値の両方）

_official = None
stats = {"calls": 0, "verified": 0, "fallbacks": 0}


def find_flattening_point(latent: torch.Tensor, target_value: float = 0.0, window_size: int = 20,
                          std_threshold: float = 0.05, mean_threshold: float = 0.1) -> int:
    """公式の find_flattening_point と同じ引数と戻り値。"""
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    total = int(latent.shape[0])
    if total <= 0 or window_size <= 0:
        return total
    stats["calls"] += 1
    a = latent.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(a).all():
        stats["fallbacks"] += 1
        return official_find_flattening_point()(latent, target_value, window_size, std_threshold, mean_threshold)
    w, n = int(window_size), int(window_size) * int(latent.shape[1])
    # 行ごとの和と 2 乗和の累積（後ろに 0 の行を window_size 個つないだもの）。
    c1 = np.zeros(total + w + 1)
    c2 = np.zeros(total + w + 1)
    c1[1:total + 1] = np.cumsum(a.sum(axis=1))
    c2[1:total + 1] = np.cumsum((a * a).sum(axis=1))
    c1[total + 1:] = c1[total]
    c2[total + 1:] = c2[total]
    mean = (c1[w:w + total] - c1[:total]) / n
    var = np.maximum((c2[w:w + total] - c2[:total]) / n - mean * mean, 0.0)
    cand = np.flatnonzero((np.sqrt(var) < std_threshold * (1 + MARGIN) + MARGIN)
                          & (np.abs(mean - target_value) < mean_threshold * (1 + MARGIN) + MARGIN))
    if cand.size == 0:
        return total
    # 候補だけを、公式と同じ式で判定する。
    pad = torch.zeros((w, latent.shape[1]), device=latent.device, dtype=latent.dtype)
    padded = torch.cat([latent, pad], dim=0)
    for i in cand.tolist():
        stats["verified"] += 1
        window = padded[i:i + w]
        window_std = window.std(unbiased=False)
        window_mean = window.mean()
        if window_std < std_threshold and torch.abs(window_mean - target_value) < mean_threshold:
            return int(i)
    return total


def install_fast_tail() -> None:
    """runtime が呼ぶ find_flattening_point を差し替える（イメージの中のコードは書き換えない）。何度呼んでもよい。"""
    global _official
    import irodori_tts.inference_runtime as IR

    if IR.find_flattening_point is not find_flattening_point:
        _official = IR.find_flattening_point
        IR.find_flattening_point = find_flattening_point


def official_find_flattening_point():
    """差し替える前の公式の関数（確かめ用）。"""
    import irodori_tts.inference_runtime as IR

    return _official if _official is not None else IR.find_flattening_point
