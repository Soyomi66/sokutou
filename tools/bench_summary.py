"""速さの記録（jsonl、1 行 1 合成）を集計して表にする。numpy だけで動く。

  python3 tools/bench_summary.py runs/bench/official.jsonl runs/bench/graphs.jsonl

各ファイルについて、1 文の合成時間（total_sec）の中央値・90 パーセンタイル・最大と、段階ごとの中央値、
文の長さ（15 字以下と 25 字以上）ごとの中央値を出す。エラーの応答の行（total_sec がない）は除き、その数を回数の横に出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    print("| 条件 | 回数 | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 生成（sample_rf か sample_meanflow） | 復号 | 電子透かし |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in args.files:
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        errors = sum("total_sec" not in r for r in rows)  # tools/bench_http.py --keep-going が記録したエラーの応答
        rows = [r for r in rows if "total_sec" in r]
        tot = np.array([r["total_sec"] for r in rows])
        chars = np.array([r.get("chars", 0) for r in rows])
        st = lambda k: np.median([r["stages"][k] for r in rows if k in r.get("stages", {})]) \
            if any(k in r.get("stages", {}) for r in rows) else float("nan")  # noqa: E731
        name = rows[0].get("engine", Path(path).stem)
        print(f"| {name} | {len(rows)}{f'（エラー {errors}）' if errors else ''} | {np.median(tot):.3f} 秒 | {np.percentile(tot, 90):.3f} 秒 | {tot.max():.3f} 秒 | "
              f"{np.median(tot[chars <= 15]) if (chars <= 15).any() else float('nan'):.3f} 秒 | "
              f"{np.median(tot[chars >= 25]) if (chars >= 25).any() else float('nan'):.3f} 秒 | "
              f"{st('sample_rf') if np.isfinite(st('sample_rf')) else st('sample_meanflow'):.3f} 秒 | {st('decode_latent'):.3f} 秒 | {st('silentcipher_watermark'):.3f} 秒 |")


if __name__ == "__main__":
    main()
