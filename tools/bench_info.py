"""冷えた状態の速さの記録（jsonl と info.json）から、起動の準備とメモリーの表を作る。numpy だけで動く。

  python3 tools/bench_info.py runs/bench/abc/*.info.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    print("| 条件 | ステップ（うち CFG） | 起動（秒） | うちグラフの記録 | うち復号の準備 | FP8 の倍率 | グラフ | 準備のあと（GB） | 96 回のあと（GB） "
          "| max_memory_allocated（GB） | 途中の記録 | 公式への切り替え | 復号の初めての形 | シード 1 の中央値（ミリ秒） | シード 2 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in sys.argv[1:]:
        p = Path(path)
        d = json.loads(p.read_text())
        rows = [json.loads(x) for x in Path(str(p).replace(".info.json", ".jsonl")).read_text(encoding="utf-8").splitlines() if x.strip()]
        med = lambda seed: 1e3 * float(np.median([r["total_sec"] for r in rows if r["seed"] == seed]))  # noqa: E731
        cal = d.get("fp8_calibrate_sec")
        print(f"| {p.name.replace('.info.json', '')} | {d.get('num_steps', 40)}（{d.get('cfg_steps', 20)}） | {d['startup_sec']:.2f} | "
              f"{d['prewarm_sec']:.2f} | {d.get('prewarm_decode_sec', float('nan')):.2f} | {'—' if cal is None else f'{cal:.2f}'} | "
              f"{d['graphs']} | {d['device_used_after_prewarm_gb']:.2f} | {d['device_used_end_gb']:.2f} | "
              f"{d['max_memory_allocated_gb']:.2f} | {d['late_captures']} | {d['fallbacks']} | {d.get('decode_cold', '—')} | "
              f"{med(1):.1f} | {med(2):.1f} |")


if __name__ == "__main__":
    main()
