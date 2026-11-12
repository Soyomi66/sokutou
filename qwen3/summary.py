"""Qwen3-TTS の記録を表にする。イメージの外の python3 で動かす（標準のライブラリだけを使う）。

  python3 qwen3/summary.py speed <表の名前>=<記録>[,<記録>...] ...
  python3 qwen3/summary.py quality <条件> ...

speed: runs/bench/qwen3/<記録>.jsonl（qwen3/bench.py と tools/bench_http.py の出力）をまとめる。1 つの行に、
  別々に測った記録（1 回目と 2 回目など）を並べられる。Qwen3-TTS はサンプリングで話す長さが変わり、実装ごとに音声の長さの
  中央値も違うので、1 文の時間に加えて「音声 1 秒あたり」（時間の合計 / 音声の長さの合計）を出す。
  時間は total_sec（bench.py は合成の時間、bench_http.py は X-Synthesis-Seconds があればその値、なければ HTTP の往復）。
  HTTP の往復（wall_sec）があれば、その音声 1 秒あたりも出す。エラーの応答（bench_http.py --keep-going）は数だけ出す。
  「上限超え」は、音声が 2.0 + 0.4 × 文字数 秒より長い本数（サーバーなら 500 を返す長さ）。
  例: python3 qwen3/summary.py speed "int8 0.6B=k2-0.6b-i8g32-1,k2-0.6b-i8g32-2" "speech.cpp 0.6B=speechcpp-0.6b-1,speechcpp-0.6b-2"
quality: runs/quality-qwen3/asr.json（tools/score_asr.py）と sim-*.json（qwen3/score_sim.py）から、条件ごとの CER と
  声の近さを出す。「悪い 2 本を除く」は、CER がいちばん高い 2 本を除いた CER（話し終えずに続いた数本で値が大きく動くため）。
  例: python3 qwen3/summary.py quality qwen3-official-0.6b qwen3-k2-0.6b-i8g32
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "runs" / "bench" / "qwen3"
QUALITY = REPO / "runs" / "quality-qwen3"


def max_seconds(chars: int) -> float:
    return 2.0 + 0.4 * chars  # qwen3/serve.py と server/server.py の式


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def speed(args: list[str]) -> None:
    print("| 条件 | 回数 | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 音声の長さの中央値 | 音声 1 秒あたり | "
          "HTTP の往復の音声 1 秒あたり | 1 フレーム（生成） | 最初の入力 | 生成 | 復号 | 上限超え | エラー |")
    print("| --- " * 16 + "|")
    for arg in args:
        label, names = arg.split("=", 1)
        rows = []
        for n in names.split(","):
            rows += [json.loads(line) for line in open(BENCH / f"{n}.jsonl", encoding="utf-8")]
        err = [r for r in rows if "total_sec" not in r]
        ok = [r for r in rows if "total_sec" in r]
        t = [r["total_sec"] for r in ok]
        a = [r["audio_sec"] for r in ok]
        short = [r["total_sec"] for r in ok if r["chars"] <= 15]
        long_ = [r["total_sec"] for r in ok if r["chars"] >= 25]
        wall = [r["wall_sec"] for r in ok if "wall_sec" in r]
        stages = [r["stages"] for r in ok if "stages" in r]
        frame = [r["stages"]["generate"] / r["frames"] * 1000 for r in ok if "stages" in r and r.get("frames")]
        over = sum(r["audio_sec"] > max_seconds(r["chars"]) for r in ok)

        def stage(k):
            v = [s[k] for s in stages if k in s]
            return f"{st.median(v) * 1000:.1f} ms" if v else "—"

        cells = [label, str(len(ok)), f"{st.median(t):.3f} 秒", f"{pct(t, 0.9):.3f} 秒", f"{max(t):.3f} 秒",
                 f"{st.median(short):.3f} 秒" if short else "—", f"{st.median(long_):.3f} 秒" if long_ else "—",
                 f"{st.median(a):.2f} 秒", f"{sum(t) / sum(a):.4f} 秒", f"{sum(wall) / sum(a):.4f} 秒" if wall else "—",
                 f"{st.median(frame):.2f} ms" if frame else "—", stage("prompt"), stage("generate"), stage("decode"),
                 str(over), str(len(err))]
        print("| " + " | ".join(cells) + " |")


def quality(conds: list[str]) -> None:
    asr = json.loads((QUALITY / "asr.json").read_text(encoding="utf-8"))
    sims: dict[str, dict[str, float]] = {}
    for f in sorted(QUALITY.glob("sim-*.json")):
        sims.update(json.loads(f.read_text(encoding="utf-8"))["sim"])
    print("| 条件 | 本数 | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除く CER | 声の近さ（平均） | 声の近さ（最小） |")
    print("| --- " * 8 + "|")
    for c in conds:
        rs = [r for r in asr if r["cond"] == c]
        if not rs:
            raise SystemExit(f"{c}: asr.json にない")
        E = sum(r["err"] for r in rs)
        N = sum(r["n"] for r in rs)
        rest = sorted(rs, key=lambda r: r["err"] / r["n"])[:-2]
        s = list(sims.get(c, {}).values())
        cells = [c, str(len(rs)), f"{100 * E / N:.2f}%", f"{sum(r['err'] > 0 for r in rs)}",
                 f"{sum(r['err'] / r['n'] > 0.3 for r in rs)}",
                 f"{100 * sum(r['err'] for r in rest) / sum(r['n'] for r in rest):.2f}%",
                 f"{st.mean(s):.3f}" if s else "—", f"{min(s):.3f}" if s else "—"]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("speed", "quality"):
        raise SystemExit(__doc__)
    {"speed": speed, "quality": quality}[sys.argv[1]](sys.argv[2:])
