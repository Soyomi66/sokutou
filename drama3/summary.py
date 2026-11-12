"""Drama 3 の記録を表にする。イメージの外の python3 で動かす（標準のライブラリだけを使う）。

  python3 drama3/summary.py speed <表の名前>=<記録>[,<記録>...] ...
  python3 drama3/summary.py quality <条件> ...

speed: runs/bench/drama3/<記録>.jsonl（drama3/bench.py と tools/bench_http.py の出力）をまとめる。サンプリングで話す長さが
  変わるので、1 文の時間に加えて「音声 1 秒あたり」（時間の合計 / 音声の長さの合計）を出す。時間は total_sec（bench.py は
  文を渡してから波形ができるまで、bench_http.py は X-Synthesis-Seconds があればその値、なければ HTTP の往復）。
  「HTTP の往復」は bench_http.py の wall_sec（speech.cpp と drama3/serve.py を同じ物差しで比べるとき）。
  「シード 2」は 2 周目（同じ 48 文の 2 回目）だけの中央値。最初の入力の CUDA Graph（--prefill-graphs）は、2 回目に出た
  区切りのときに記録するので、その文に記録の時間が入る（「記録した文」がその数）。
  例: python3 drama3/summary.py speed "公式=official-1" "エンジン=kernels-1" "speech.cpp Q4_K=speechcpp-q4k-1"
quality: runs/quality-drama3/asr.json（tools/score_asr.py）と sim-*.json（drama3/score_sim.py）から、条件ごとの CER と
  声の近さを出す。「悪い 2 本を除く」は、CER がいちばん高い 2 本を除いた CER。
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "runs" / "bench" / "drama3"
QUALITY = REPO / "runs" / "quality-drama3"


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def speed(args: list[str]) -> None:
    print("| 条件 | 回数 | 中央値 | シード 2 の中央値 | 90 パーセンタイル | 最大 | 音声の長さの中央値 | 音声 1 秒あたり | "
          "HTTP の往復の中央値 | 1 フレーム | 最初の入力 | 復号 | 記録した文 | エラー |")
    print("| --- " * 14 + "|")
    for arg in args:
        label, names = arg.split("=", 1)
        rows = []
        for n in names.split(","):
            rows += [json.loads(line) for line in open(BENCH / f"{n}.jsonl", encoding="utf-8")]
        err = [r for r in rows if "total_sec" not in r]
        ok = [r for r in rows if "total_sec" in r]
        t = [r["total_sec"] for r in ok]
        t2 = [r["total_sec"] for r in ok if r.get("seed") == 2]
        a = [r["audio_sec"] for r in ok]
        stages = [r["stages"] for r in ok if "stages" in r]
        wall = [r["wall_sec"] for r in ok if "wall_sec" in r]
        # 1 フレーム: 生成の時間から最初の入力を引き、フレームの数で割る
        frame = [(r["stages"]["generate"] - r["stages"].get("prefill", 0.0)) / r["frames"] * 1000
                 for r in ok if "stages" in r and r.get("frames")]

        def stage(k):
            v = [s[k] for s in stages if k in s]
            return f"{st.median(v) * 1000:.0f} ms" if v else "—"

        cells = [label, str(len(ok)), f"{st.median(t):.3f} 秒", f"{st.median(t2):.3f} 秒" if t2 else "—",
                 f"{pct(t, 0.9):.3f} 秒", f"{max(t):.3f} 秒", f"{st.median(a):.2f} 秒", f"{sum(t) / sum(a):.4f} 秒",
                 f"{st.median(wall):.3f} 秒" if wall else "—", f"{st.median(frame):.2f} ms" if frame else "—", stage("prefill"), stage("decode"),
                 str(sum(r.get("prefill_captures", 0) > 0 for r in ok)), str(len(err))]
        print("| " + " | ".join(cells) + " |")


def quality(conds: list[str]) -> None:
    asr = json.loads((QUALITY / "asr.json").read_text(encoding="utf-8"))
    sims: dict[str, dict[str, dict[str, float]]] = {}
    for f in sorted(QUALITY.glob("sim-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for kind in ("reference", "centroid"):
            for c, v in d.get(kind, {}).items():
                sims.setdefault(c, {})[kind] = v
    print("| 条件 | 本数 | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除く CER | 参照との近さ（平均） | "
          "公式の重心との近さ（平均） |")
    print("| --- " * 8 + "|")
    for c in conds:
        rs = [r for r in asr if r["cond"] == c]
        if not rs:
            raise SystemExit(f"{c}: asr.json にない")
        E = sum(r["err"] for r in rs)
        N = sum(r["n"] for r in rs)
        rest = sorted(rs, key=lambda r: r["err"] / r["n"])[:-2]
        s = sims.get(c, {})

        def mean(kind):
            v = list(s.get(kind, {}).values())
            return f"{st.mean(v):.3f}" if v else "—"

        cells = [c, str(len(rs)), f"{100 * E / N:.2f}%", f"{sum(r['err'] > 0 for r in rs)}",
                 f"{sum(r['err'] / r['n'] > 0.3 for r in rs)}",
                 f"{100 * sum(r['err'] for r in rest) / sum(r['n'] for r in rest):.2f}%", mean("reference"), mean("centroid")]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("speed", "quality"):
        raise SystemExit(__doc__)
    {"speed": speed, "quality": quality}[sys.argv[1]](sys.argv[2:])
