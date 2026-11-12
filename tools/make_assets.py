"""`assets/` の画像（ロゴの記号のアイコン、1:1 のアイコン、速さの棒グラフ）を書き出す。Python の標準ライブラリだけで動く。

  python3 tools/make_assets.py

書き出すのは次の 5 つ。
- `assets/icon.svg`: README の先頭のアイコン。記号だけで、地は透明
- `assets/avatar.svg`、`assets/avatar.png`（1024×1024）: 1:1 のアイコン。白い地に記号を置き、円に切り抜かれても欠けない余白を取る
- `assets/speed-light.svg`、`assets/speed-dark.svg`: README の「速さ」の棒グラフ

グラフの値は `docs/results.md` の「速さ」の表（v4.1-Small、MF 版、v4-Large）の中央値から読む。
そのあと、README のグラフの alt 文が同じ値になっているかを確かめ、違えば正しい alt 文を出して 1 で終わる。

ロゴは、鉤括弧「」で S を組んだ記号。色は朱 #e0482a で、GitHub のライトの地（#ffffff）で 4.09:1、
ダークの地（#0d1117）で 4.62:1 なので、アイコンは 1 枚で両方に使う。グラフで朱を使うのは sokutou の棒だけ。
ロゴと名前を組んだ形とソーシャルプレビューの元は Figma にある（`TASKS.md` の「README の画像」）。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import zlib
from pathlib import Path

SHU = "#e0482a"

# Figma の「sokutou / Mark」と同じ形（64 の格子）。鉤括弧「と」を、2 つずつの長方形で描く
MARK_RECTS = [
    (12, 8, 52, 18), (12, 18, 22, 36),   # 「
    (12, 46, 52, 56), (42, 28, 52, 46),  # 」
]
MARK = '<path d="M12 8H52V18H22V36H12Z"/><path d="M52 56H12V46H42V28H52Z"/>'

ICON = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="4 4 56 56" fill="{SHU}">
  {MARK}
</svg>
"""

# 1:1 のアイコン。80 の正方形の真ん中（32, 32）に記号を置く。記号の高さは 6 割で、
# 記号の角から中心までは 31.2 なので、半径 40 の円に切り抜いても 8 割の内側に収まる
AVATAR_VIEW = (-8, 80)
AVATAR_PX = 1024
AVATAR = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{AVATAR_VIEW[0]} {AVATAR_VIEW[0]} {AVATAR_VIEW[1]} {AVATAR_VIEW[1]}">
  <rect x="{AVATAR_VIEW[0]}" y="{AVATAR_VIEW[0]}" width="{AVATAR_VIEW[1]}" height="{AVATAR_VIEW[1]}" fill="#ffffff"/>
  <g fill="{SHU}">{MARK}</g>
</svg>
"""


def avatar_png(size: int = AVATAR_PX) -> bytes:
    """1:1 のアイコンの PNG。記号は長方形だけなので、画素ごとに覆う面積を正確に求めて塗る（縁は中間の色になる）。"""
    origin, span = AVATAR_VIEW
    scale = size / span

    def cover(a: float, b: float) -> list[float]:
        """画素 i（[i, i+1)）と区間 [a, b) の重なりの長さ。"""
        out = [0.0] * size
        for i in range(max(0, int(a)), min(size, int(b) + 1)):
            out[i] = max(0.0, min(b, i + 1) - max(a, i))
        return out

    rects = [(cover((x0 - origin) * scale, (x1 - origin) * scale), cover((y0 - origin) * scale, (y1 - origin) * scale))
             for x0, y0, x1, y1 in MARK_RECTS]
    fg = tuple(int(SHU[i:i + 2], 16) for i in (1, 3, 5))
    lut = [bytes(round(255 + (c - 255) * k / 255) for c in fg) for k in range(256)]
    white = lut[0] * size
    rows = []
    for y in range(size):
        active = [(cx, cy[y]) for cx, cy in rects if cy[y] > 0]
        if not active:
            rows.append(white)
            continue
        cov = [0.0] * size
        for cx, wy in active:
            for x, wx in enumerate(cx):
                if wx:
                    cov[x] += wx * wy
        rows.append(b"".join(lut[min(255, round(c * 255))] for c in cov))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + r for r in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

# docs/results.md の見出し、グラフの見出し、読むステップ数
SECTIONS = [
    ("## 速さ", "v4.1-Small（20 ステップ）", 20),
    ("### MF 版", "v4.1-Small-MF（4 ステップ）", 4),
    ("### v4-Large", "v4-Large（20 ステップ）", 20),
]
# グラフでの名前と、表の「実装」の列の書き出し
IMPLS = [("sokutou", "sokutou（"), ("speech.cpp", "speech.cpp "), ("公式の実装", "公式の実装（")]

FONT = ("system-ui, -apple-system, 'Segoe UI', 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', "
        "'Noto Sans JP', 'Noto Sans CJK JP', 'Yu Gothic UI', Meiryo, sans-serif")

# 文字は GitHub の文字の色。灰の棒は 3:1 以上（ライト 3.45、ダーク 3.57）
THEMES = {
    "light": dict(text="#1f2328", text2="#59636e", other="#818b98", base="#d1d9e0"),
    "dark": dict(text="#f0f6fc", text2="#9198a1", other="#656c76", base="#3d444d"),
}


def read_panels(results: str) -> list[tuple[str, list[tuple[str, float]]]]:
    """docs/results.md の 3 つの表から、実装ごとの中央値（秒）を読む。"""
    lines = results.splitlines()
    panels = []
    for heading, title, steps in SECTIONS:
        if heading not in lines:
            sys.exit(f"docs/results.md に見出し「{heading}」がない")
        values: dict[str, float] = {}
        for line in lines[lines.index(heading) + 1:]:
            if line.startswith("#"):
                break
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[1] != str(steps):
                continue
            for name, prefix in IMPLS:
                if cells[0].startswith(prefix) and name not in values:
                    values[name] = float(cells[2].removesuffix("秒").strip())
        missing = [name for name, _ in IMPLS if name not in values]
        if missing:
            sys.exit(f"「{heading}」の表に {steps} ステップの行がない: {'、'.join(missing)}")
        panels.append((title, [(name, values[name]) for name, _ in IMPLS]))
    return panels


def alt_text(panels: list[tuple[str, list[tuple[str, float]]]]) -> str:
    parts = []
    for i, (title, rows) in enumerate(panels):
        if i == 0:
            body = "、".join(f"{name} {v:.3f} 秒" for name, v in rows)
        else:
            body = "、".join(f"{v:.3f} 秒" for _, v in rows)
        parts.append(f"{title}は {body}")
    return "1 文の合成にかかる時間の中央値。" + "。".join(parts)


def bar_path(x0: float, y0: float, x1: float, h: float, r: float = 4) -> str:
    """左（基準線）は角、右（値の端）は 4px の丸み。"""
    r = min(r, (x1 - x0) / 2, h / 2)
    y1 = y0 + h
    return (f'M{x0:.2f} {y0:.2f}H{x1 - r:.2f}A{r} {r} 0 0 1 {x1:.2f} {y0 + r:.2f}'
            f'V{y1 - r:.2f}A{r} {r} 0 0 1 {x1 - r:.2f} {y1:.2f}H{x0:.2f}Z')


def chart(theme: str, panels: list[tuple[str, list[tuple[str, float]]]]) -> str:
    t = THEMES[theme]
    W, pad = 520, 16
    label_right, bar_left = 106, 116
    bar_max = W - pad - 72 - bar_left
    row, thick = 32, 16
    panel_h, panel_gap = 24 + 3 * row, 16
    top = 88
    H = top + len(panels) * (panel_h + panel_gap) + 44

    desc = "。".join(
        f"{title}: " + "、".join(f"{name} {v:.3f} 秒" for name, v in rows) for title, rows in panels)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">',
        '<title id="t">1 文の合成にかかる時間（中央値）</title>',
        f'<desc id="d">{desc}。</desc>',
        f'<text x="{pad}" y="28" font-size="19" font-weight="600" fill="{t["text"]}">1 文の合成にかかる時間（中央値）</text>',
        f'<text x="{pad}" y="56" font-size="14" fill="{t["text2"]}">RTX 6000 Ada で 192 回測った中央値。短いほど速い</text>',
    ]
    y = top
    for title, rows in panels:
        out.append(f'<text x="{pad}" y="{y + 4}" font-size="15" font-weight="600" fill="{t["text"]}">{title}</text>')
        vmax = max(v for _, v in rows)
        y_rows = y + 20
        for i, (name, v) in enumerate(rows):
            yb = y_rows + i * row
            cy = yb + thick / 2
            hero = name == "sokutou"
            ink = t["text"] if hero else t["text2"]
            fw = ' font-weight="600"' if hero else ""
            x1 = bar_left + bar_max * v / vmax
            out.append(f'<text x="{label_right}" y="{cy:.1f}" dy="0.35em" text-anchor="end" font-size="15" '
                       f'fill="{ink}"{fw}>{name}</text>')
            out.append(f'<path d="{bar_path(bar_left, yb, x1, thick)}" fill="{SHU if hero else t["other"]}"/>')
            out.append(f'<text x="{x1 + 8:.2f}" y="{cy:.1f}" dy="0.35em" font-size="15" '
                       f'style="font-variant-numeric:tabular-nums" fill="{ink}"{fw}>{v:.3f} 秒</text>')
        out.append(f'<line x1="{bar_left - 0.5}" x2="{bar_left - 0.5}" y1="{y_rows - 4}" '
                   f'y2="{y_rows + 2 * row + thick + 4}" stroke="{t["base"]}" stroke-width="1"/>')
        y += panel_h + panel_gap
    out.append(f'<text x="{pad}" y="{y + 8}" font-size="13" fill="{t["text2"]}">'
               'speech.cpp は v0.10.1（Vulkan、F16）で、HTTP を含む値</text>')
    out.append(f'<text x="{pad}" y="{y + 28}" font-size="13" fill="{t["text2"]}">'
               '棒の目盛りはモデルごとに違う</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    repo: Path = args.repo

    panels = read_panels((repo / "docs/results.md").read_text(encoding="utf-8"))
    (repo / "assets/icon.svg").write_text(ICON, encoding="utf-8")
    (repo / "assets/avatar.svg").write_text(AVATAR, encoding="utf-8")
    (repo / "assets/avatar.png").write_bytes(avatar_png())
    for theme in THEMES:
        (repo / f"assets/speed-{theme}.svg").write_text(chart(theme, panels), encoding="utf-8")
    for title, rows in panels:
        print(title + ": " + "、".join(f"{name} {v:.3f} 秒" for name, v in rows))

    readme = (repo / "README.md").read_text(encoding="utf-8")
    m = re.search(r'<img src="assets/speed-light\.svg"[^>]*\balt="([^"]*)"', readme)
    expected = alt_text(panels)
    if m is None:
        sys.exit("README.md に assets/speed-light.svg の <img> がない")
    if m.group(1) != expected:
        print("README.md のグラフの alt 文が、表の値と合わない。次の文に直す:", file=sys.stderr)
        print(expected, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
