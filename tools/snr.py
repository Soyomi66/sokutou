"""参照（runs/ref/<tag>/）と候補（runs/cand/<name>/）を段階ごとに SNR（dB）で比べる。numpy だけで動く。

SNR = 10 log10( Σ ref² / Σ (ref − cand)² )。誤差が 0 なら inf。最大絶対誤差も出す。
文と話者の状態とその KV は、マスクが有効なトークンだけを比べる（パディングの行は使われないので、
パディングをなくしたエンジンでも比べられるようにする）。caption は空なので比べない。

比べるもの:
- teacher forcing（参照の入力を入れた結果）: 条件のエンコード、KV、各ステップの v（CFG の 3 つ分と合成後）、
  ステップ 0 と 20 の各層の出力、復号、電子透かし
- 通し（最初のノイズから最後まで）: 各ステップの x、最後の x、復号、電子透かし、24kHz の出力

使い方: python3 tools/snr.py runs/ref/gpu runs/cand/official-cpu [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(p: Path) -> dict[str, np.ndarray]:
    if not p.exists():
        return {}
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def snr_db(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if ref.shape != cand.shape:
        n = min(ref.size, cand.size)
        if ref.ndim == 1 or ref.squeeze().ndim == 1:  # 音声の長さが違うときは短いほうに合わせる
            ref, cand = ref.reshape(-1)[:n], cand.reshape(-1)[:n]
        else:
            raise ValueError(f"shape mismatch {ref.shape} vs {cand.shape}")
    err = float(np.sum((ref - cand) ** 2))
    sig = float(np.sum(ref ** 2))
    maxabs = float(np.max(np.abs(ref - cand))) if ref.size else 0.0
    return (math.inf if err == 0 else 10 * math.log10(sig / err)), maxabs


def valid_rows(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(B, S, ...) の状態から、マスクが有効な行だけを取り出す。"""
    return state[mask.astype(bool)]


def compare_case(ref: Path, cand: Path) -> list[dict]:
    rows: list[dict] = []

    def add(stage: str, r, c, note: str = "") -> None:
        if r is None or c is None:
            return
        s, m = snr_db(r, c)
        rows.append({"stage": stage, "snr_db": s, "maxabs": m, "note": note})

    rc, cc = load(ref / "cond.npz"), load(cand / "cond.npz")
    if rc and cc:
        tm = rc["text_mask"]
        add("cond.text_state", valid_rows(rc["text_state"], tm), valid_rows(cc["text_state"], cc["text_mask"]),
            f"有効 {int(tm.sum())} トークン")
        sm = rc["speaker_mask"]
        add("cond.speaker_state", valid_rows(rc["speaker_state"], sm), valid_rows(cc["speaker_state"], cc["speaker_mask"]),
            f"有効 {int(sm.sum())} トークン")
    rk, ck = load(ref / "kv.npz"), load(cand / "kv.npz")
    names = {0: "k_text", 1: "v_text", 2: "k_speaker", 3: "v_speaker"}
    for k in sorted(rk):
        if k in ck:
            layer, j = k.split("_")[1], int(k.split("_")[2])
            if j not in names:
                continue  # caption の KV は空なので比べない
            key = "text_mask" if j < 2 else "speaker_mask"
            # 候補がパディングを持たない形で書いたときは、候補自身のマスクで有効な行を取り出す。
            cmask = cc[key] if cc and cc[key].shape[1] == ck[k].shape[1] else rc[key]
            add(f"kv.{layer}.{names[j]}", valid_rows(rk[k], rc[key]), valid_rows(ck[k], cmask))
    rs, cs = load(ref / "steps.npz"), load(cand / "steps_tf.npz")
    if rs and cs:
        for key in ("v_raw_cfg", "v_raw_nocfg", "v"):
            if len(rs[key]) == 0:  # MF 版には CFG のステップがない
                continue
            per = [snr_db(rs[key][i], cs[key][i])[0] for i in range(len(rs[key]))]
            fin = [p for p in per if math.isfinite(p)]
            add(f"tf.{key}", rs[key], cs[key],
                f"ステップごとの最小 {min(per):.1f} dB" + (f"、中央値 {np.median(fin):.1f} dB" if fin else ""))
    for step in (0, 20):
        rb, cb = load(ref / f"blocks_s{step:02d}.npz"), load(cand / f"blocks_s{step:02d}.npz")
        for k in sorted(rb):
            if k in cb:
                add(f"tf.s{step:02d}.{k}", rb[k], cb[k])
    ra, ca = load(ref / "audio.npz"), load(cand / "audio_tf.npz")
    if ra and ca:
        add("tf.decode_out", ra["decode_out"], ca["decode_out"])
        add("tf.wm_out", ra["wm_out"], ca["wm_out"])
    cf = load(cand / "free.npz")
    if rs and cf:
        if "x" in cf:  # 各ステップの x を書かない候補もある（CUDA Graph など）
            per = [snr_db(rs["x"][i], cf["x"][i])[0] for i in range(1, len(rs["x"]))]
            n = len(per)
            picks = sorted({i for i in (1, n // 4, n // 2, 3 * n // 4, n) if 1 <= i <= n})
            add("free.x_steps", rs["x"], cf["x"], f"ステップ 1〜{n} の SNR: " + ", ".join(
                f"{i}:{per[i - 1]:.0f}" for i in picks))
        add("free.x_final", rs["x_final"], cf["x_final"])
        add("free.decode_out", ra["decode_out"], cf["decode_out"])
        add("free.wm_out", ra["wm_out"], cf["wm_out"])
        add("free.out24k", ra["out24k"], cf["out24k"],
            f"長さ {ra['out24k'].size} / {cf['out24k'].size}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--json")
    ap.add_argument("--stages", default="", help="この文字列を含む段階だけ表示する（カンマ区切り）")
    args = ap.parse_args()
    ref_root, cand_root = Path(args.ref), Path(args.cand)
    result = {}
    for ref_case in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        cand_case = cand_root / ref_case.name
        if not cand_case.exists():
            continue
        result[ref_case.name] = compare_case(ref_case, cand_case)
    # ケースをまたいで段階ごとの最小 SNR を出す。
    stages: dict[str, list[dict]] = {}
    for rows in result.values():
        for r in rows:
            stages.setdefault(r["stage"], []).append(r)
    filt = [s for s in args.stages.split(",") if s]
    print(f"{'stage':40s} {'min SNR dB':>11s} {'med SNR dB':>11s} {'max |err|':>11s}  note（最初のケース）")
    for stage, rs in stages.items():
        if filt and not any(f in stage for f in filt):
            continue
        snrs = [r["snr_db"] for r in rs]
        print(f"{stage:40s} {min(snrs):11.1f} {float(np.median(snrs)):11.1f} {max(r['maxabs'] for r in rs):11.3g}  "
              f"{rs[0]['note']}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
