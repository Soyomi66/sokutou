"""合成音声の前後の無音と、文末の無音の後ろに付く短いノイズを切り落とす。

Irodori-TTS は、予測した長さが発話より長いと、文末に 0.1〜0.3 秒の余計な声（「あっ」など）や
1 秒前後の無音を足すことがある。無音のあとに続く短い音だけを落とし、発話の途中は切らない。
"""

from __future__ import annotations

import numpy as np


def trim_audio(
    audio: np.ndarray,
    sr: int,
    *,
    floor_db: float = -45.0,
    below_peak_db: float = 35.0,
    frame_sec: float = 0.01,
    merge_gap_sec: float = 0.12,
    tail_gap_sec: float = 0.2,
    tail_max_sec: float = 0.3,
    pad_head_sec: float = 0.05,
    pad_tail_sec: float = 0.12,
) -> np.ndarray:
    """有音区間を見つけ、前後の無音と文末の短いノイズを除いた範囲を返す。"""
    hop = max(1, int(sr * frame_sec))
    n = len(audio) // hop
    if n == 0:
        return audio
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    threshold = max(floor_db, float(rms_db.max()) - below_peak_db)
    voiced = rms_db > threshold
    if not voiced.any():
        return audio

    # 有音のフレームを区間にまとめる（短い無音は同じ区間とみなす）。
    segments: list[list[int]] = []
    merge = int(merge_gap_sec / frame_sec)
    for i in np.flatnonzero(voiced):
        if segments and i - segments[-1][1] <= merge:
            segments[-1][1] = int(i)
        else:
            segments.append([int(i), int(i)])

    # 文末の、無音のあとに続く短い区間はノイズとして落とす。
    gap = int(tail_gap_sec / frame_sec)
    short = int(tail_max_sec / frame_sec)
    while len(segments) > 1:
        last, prev = segments[-1], segments[-2]
        if last[1] - last[0] + 1 <= short and last[0] - prev[1] >= gap:
            segments.pop()
        else:
            break

    start = max(0, segments[0][0] * hop - int(pad_head_sec * sr))
    end = min(len(audio), (segments[-1][1] + 1) * hop + int(pad_tail_sec * sr))
    return audio[start:end]
