"""qwen3/bench.py と tools/bench_http.py が保存した WAV を、tools/score_asr.py と qwen3/score_sim.py が読む一覧
（runs/quality-qwen3/rows-<条件>.json）にする。イメージの外の python3 で動かす（標準のライブラリだけを使う）。

  python3 qwen3/quality_rows.py <条件>=<WAV のディレクトリ>/<名前> [<ディレクトリ>/<名前> ...] ...

<ディレクトリ> は runs/bench/qwen3/ の下、<名前> は WAV の名前の前半（<名前>_s<シード>_<番号>.wav）。
1 つの条件に、別々に測ったものを並べられる（シード 1, 2 と 3〜6 を別に測ったときなど）。例:

  python3 qwen3/quality_rows.py qwen3-official-0.6b=official-0.6b-wav/official-0.6b-1 \\
      qwen3-official-0.6b-s6=official-0.6b-wav/official-0.6b-1,official-0.6b-s36-wav/official-0.6b-s3to6
"""

from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "runs" / "bench" / "qwen3"
OUT = REPO / "runs" / "quality-qwen3"


def main() -> None:
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        cond, srcs = arg.split("=", 1)
        rows = []
        for src in srcs.split(","):
            d, stem = src.rsplit("/", 1)
            pat = re.compile(re.escape(stem) + r"_s(\d+)_(\d+)\.wav$")
            for f in sorted((BENCH / d).glob(f"{stem}_s*_*.wav")):
                m = pat.match(f.name)
                if not m:
                    continue
                seed, i = int(m.group(1)), int(m.group(2))
                with wave.open(str(f)) as w:
                    sec = w.getnframes() / w.getframerate()
                rows.append({"cond": cond, "i": i, "seed": seed, "file": f"../bench/qwen3/{d}/{f.name}",
                             "text": sentences[i]["text"], "seconds": sentences[i].get("seconds"), "audio_sec": sec})
        rows.sort(key=lambda r: (r["seed"], r["i"]))
        keys = {(r["seed"], r["i"]) for r in rows}
        if len(keys) != len(rows):
            raise SystemExit(f"{cond}: 同じシードと番号の WAV が 2 つ以上ある")
        (OUT / f"rows-{cond}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(cond, len(rows), "本（シード", sorted({r["seed"] for r in rows}), "）")


if __name__ == "__main__":
    main()
