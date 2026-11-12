"""聞き比べのページ（runs/quality/listen.html）を作る。

「今のエンジン」（fused、BF16、40 ステップ、復号を決まった長さで動かす、透かしなし）と、出力が変わるそれぞれの版を、
同じ文・同じシードで並べる。「今の本番」（公式の実装、fp32、40 ステップ、透かしあり）も並べる。
列が多いので、比べる版をページの上で選ぶ。音声は MP3 にして HTML に埋め込む（1 つのファイルで開ける）。
参照音声の派生物なので、出力は runs/ の下にだけ置く。

  python3 tools/make_listen.py                # 48 文 × シード 1, 2 のすべて → runs/quality/listen.html
  python3 tools/make_listen.py --tops 8       # 版ごとに、今のエンジンより声の近さが下がった 8 組 → listen-top8-<条件>.html
  python3 tools/make_listen.py --focus fused-bf16-nowm-bdec-s20 --limit 8 --out runs/quality/x.html

合成時間は、runs/bench/abc/ の冷えた状態の記録（48 文 × シード 1, 2 を 2 回）の中央値を読む。
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
Q = REPO / "runs" / "quality"
BENCH = REPO / "runs" / "bench" / "abc"
PROD, CURRENT = "official", "fused-bf16-nowm-bdec"
E = "fused-bf16-nowm"
# 条件ごとの説明。group: prod（今の本番）、current（今のエンジン）、new（今回の版）、old（前に聞いた版）。
# speed は、記録から読めない条件の合成時間（前の計測の値）。
CONDS = [
    dict(key=PROD, group="prod", title="今の本番", prec="fp32", steps=40, decode="元の長さ・fp32", wm="あり",
         what="公式の実装", speed="0.775 秒（前の計測、透かしあり）"),
    dict(key=CURRENT, group="current", title="今のエンジン", prec="BF16", steps=40, decode="決まった長さ・fp32", wm="なし",
         what="fused + --bucket-decode"),
    dict(key=f"{E}-bdec-s20", group="new", title="20 ステップ", prec="BF16", steps=20, decode="決まった長さ・fp32", wm="なし",
         what="今のエンジンのステップ数を 20 に（CFG は前半 10 ステップ）"),
    dict(key=f"{E}-bdec-fp8nocfg", group="new", title="FP8（CFG なしのステップ）", prec="BF16 + FP8", steps=40,
         decode="決まった長さ・fp32", wm="なし", what="後半 20 ステップの行列積だけ FP8"),
    dict(key=f"{E}-bdec-fp8cfg", group="new", title="FP8（CFG ありのステップ）", prec="BF16 + FP8", steps=40,
         decode="決まった長さ・fp32", wm="なし", what="前半 20 ステップの行列積だけ FP8"),
    dict(key=f"{E}-bdec-fp8", group="new", title="FP8（全ステップ）", prec="FP8", steps=40, decode="決まった長さ・fp32", wm="なし",
         what="40 ステップすべての行列積を FP8"),
    dict(key=f"{E}-bdec-fp8nocfg-s20", group="new", title="20 ステップ + FP8（CFG なし）", prec="BF16 + FP8", steps=20,
         decode="決まった長さ・fp32", wm="なし", what="20 ステップで、後半 10 ステップの行列積だけ FP8"),
    dict(key=f"{E}-bdec-fp8-s20", group="new", title="20 ステップ + FP8（全ステップ）", prec="FP8", steps=20,
         decode="決まった長さ・fp32", wm="なし", what="20 ステップすべての行列積を FP8"),
    dict(key=f"{E}-hdec-bdec", group="new", title="復号 float16", prec="BF16", steps=40, decode="決まった長さ・float16", wm="なし",
         what="今のエンジンの復号を float16 に（--half-decode）"),
    dict(key="official-s20", group="new", title="公式の 20 ステップ", prec="fp32", steps=20, decode="元の長さ・fp32", wm="あり",
         what="公式の実装のステップ数を 20 に", speed="0.44 秒（作者の前の計測）"),
    dict(key="fused-bf16", group="old", title="前の fused", prec="BF16", steps=40, decode="元の長さ・fp32", wm="あり",
         what="前のページの「カーネルをまとめた版」", speed="—"),
    dict(key="fused-bf16-hdec", group="old", title="前の fused + 復号 float16", prec="BF16", steps=40, decode="元の長さ・float16",
         wm="あり", what="前のページの「復号 fp16」", speed="—"),
    dict(key="bucket-bf16", group="old", title="前の bucket", prec="BF16", steps=40, decode="元の長さ・fp32", wm="あり",
         what="前のページの「本番に使う版」", speed="—"),
    dict(key="fast-bf16", group="old", title="前の fast BF16", prec="BF16", steps=40, decode="元の長さ・fp32", wm="あり",
         what="前のページの「前に聞いた版」", speed="—"),
    dict(key="fast-tf32", group="old", title="前の fast TF32", prec="TF32", steps=40, decode="元の長さ・fp32", wm="あり",
         what="前のページの TF32", speed="—"),
]


def mp3_b64(path: Path, bitrate: str) -> str:
    mp3 = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-b:a", bitrate, "-f", "mp3", "-"],
        check=True, capture_output=True,
    ).stdout
    return base64.b64encode(mp3).decode()


def cold_median(cond: str) -> str | None:
    """runs/bench/abc/ の記録から、1 文の合成時間の中央値（回ごとの中央値の中央値）を読む。"""
    parts = cond.split("-")
    if len(parts) < 2 or not BENCH.exists():
        return None
    name = "-".join(parts[:2] + ["cold"] + parts[2:])
    meds = []
    for f in sorted(BENCH.glob(f"{name}-cc-*.jsonl")):
        tot = [json.loads(x)["total_sec"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        meds.append(statistics.median(tot))
    return f"{statistics.median(meds):.3f} 秒" if meds else None


def load(conds: list[dict]) -> tuple[list[dict], dict]:
    """戻り値: (行の一覧, 条件ごとの集計)。行は {i, seed, text, v: {条件: {...}}}。"""
    sim = json.loads((Q / "sim.json").read_text(encoding="utf-8"))["sim"]
    asr = {r["file"]: r for r in json.loads((Q / "asr.json").read_text(encoding="utf-8"))}
    meta = {c["key"]: {(r["i"], r["seed"]): r for r in json.loads((Q / f"rows-{c['key']}.json").read_text(encoding="utf-8"))}
            for c in conds}
    rows = []
    for (i, seed), base in sorted(meta[CURRENT].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        v = {}
        for c in conds:
            r = meta[c["key"]][(i, seed)]
            a = asr[r["file"]]
            v[c["key"]] = {"file": r["file"], "sim": round(float(sim[r["file"]]), 3), "err": int(a["err"]), "n": int(a["n"]),
                           "hyp": a["hyp"], "wm": None if not r.get("wm_24k") else bool(r["wm_24k"]["ok"]),
                           "sec": round(float(r["audio_sec"]), 2)}
        rows.append({"i": i, "seed": seed, "text": base["text"], "v": v})
    stats = {}
    for c in conds:
        k = c["key"]
        vs = [r["v"][k] for r in rows]
        d = [r["v"][k]["sim"] - r["v"][CURRENT]["sim"] for r in rows]
        stats[k] = {"cer": 100 * sum(x["err"] for x in vs) / sum(x["n"] for x in vs), "bad": sum(x["err"] > 0 for x in vs),
                    "sim": sum(x["sim"] for x in vs) / len(vs), "sim_min": min(x["sim"] for x in vs),
                    "d_mean": sum(d) / len(d), "d_min": min(d), "d_05": sum(x < -0.05 for x in d), "n": len(vs)}
    return rows, stats


STYLE = """
:root { color-scheme: light dark;
  --ink: var(--foreground, #16181d); --ink-2: var(--muted-foreground, #5d6270); --rule: var(--border, #d9dce3);
  --surface: var(--card, #f4f5f8); --accent: var(--primary, #2457d6); --accent-ink: var(--primary-foreground, #fff);
  --warn: var(--warning, #b25e00); --r: var(--radius, 8px); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ink: var(--foreground, #eceef2); --ink-2: var(--muted-foreground, #9aa0ad); --rule: var(--border, #343843);
  --surface: var(--card, #1b1e25); --accent: var(--primary, #6b93ff); --accent-ink: var(--primary-foreground, #0d1020);
  --warn: var(--warning, #f0a24a); } }
:root[data-theme="dark"] { --ink: var(--foreground, #eceef2); --ink-2: var(--muted-foreground, #9aa0ad);
  --rule: var(--border, #343843); --surface: var(--card, #1b1e25); --accent: var(--primary, #6b93ff);
  --accent-ink: var(--primary-foreground, #0d1020); --warn: var(--warning, #f0a24a); }
body { margin: 0; color: var(--ink); font-family: var(--font-sans, system-ui, sans-serif); font-size: 14px; line-height: 1.55; }
.page { max-width: 980px; margin: 0 auto; padding: 4px 0 24px; }
h1 { font-size: 18px; margin: 0 0 6px; font-weight: 650; }
.lead { margin: 0 0 14px; color: var(--ink-2); font-size: 13px; }
.tablewrap { overflow-x: auto; margin-bottom: 14px; border: 1px solid var(--rule); border-radius: var(--r); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 9px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--ink-2); background: var(--surface); }
th:first-child, td:first-child { text-align: left; white-space: normal; min-width: 150px; }
td small { display: block; color: var(--ink-2); font-size: 11.5px; }
tr:last-child td { border-bottom: 0; }
tr.cur td { background: color-mix(in srgb, var(--accent) 9%, transparent); }
tr.sep td { border-top: 2px solid var(--rule); }
td.bad { color: var(--warn); }
.picker { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 4px 14px; margin: 0 0 10px; padding: 10px 12px;
  border: 1px solid var(--rule); border-radius: var(--r); }
.picker legend { font-size: 12.5px; color: var(--ink-2); padding: 0 4px; }
.picker label { display: flex; align-items: baseline; gap: 6px; font-size: 13px; }
.picker label.fixed { color: var(--ink-2); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; padding: 10px 0; border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule); margin-bottom: 6px; position: sticky; top: 0; z-index: 3;
  background: var(--background, Canvas); }
.bar label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
.bar select { font: inherit; font-size: 13px; color: var(--ink); background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; padding: 3px 6px; }
.bar .count { color: var(--ink-2); font-size: 12.5px; margin-left: auto; }
.row { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px 16px; padding: 12px 0;
  border-bottom: 1px solid var(--rule); }
.text { font-size: 14.5px; }
.meta { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.reveal { margin-top: 6px; font: inherit; font-size: 12px; color: var(--accent); background: none; border: 1px solid var(--rule);
  border-radius: 6px; padding: 2px 8px; cursor: pointer; }
.vers { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.v { border: 1px solid var(--rule); border-radius: var(--r); padding: 8px; background: var(--surface); min-width: 0; }
.v.cur { border-color: var(--accent); border-width: 2px; padding: 7px; }
.v .delta { font-variant-numeric: tabular-nums; }
.v .name { font-size: 12.5px; font-weight: 600; margin-bottom: 6px; display: flex; flex-direction: column; gap: 0; }
.v .name small { font-weight: 400; color: var(--ink-2); }
.play { width: 100%; white-space: nowrap; display: flex; align-items: center; gap: 8px; font: inherit; font-size: 13px; color: var(--ink);
  background: none; border: 1px solid var(--rule); border-radius: 6px; padding: 5px 8px; cursor: pointer; position: relative;
  overflow: hidden; text-align: left; }
.play:hover { border-color: var(--accent); }
.play.on { border-color: var(--accent); }
.play .fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; background: var(--accent); opacity: .16; pointer-events: none; }
.play svg { width: 14px; height: 14px; flex: none; fill: currentColor; }
.play .t { color: var(--ink-2); font-size: 12px; margin-left: auto; font-variant-numeric: tabular-nums; }
.stats { margin-top: 6px; font-size: 11.5px; color: var(--ink-2); line-height: 1.45; }
.stats .bad { color: var(--warn); }
.stats span { white-space: nowrap; }
.hyp { margin-top: 2px; font-size: 11.5px; color: var(--ink-2); overflow-wrap: anywhere; }
body.blind .row:not(.shown) .stats, body.blind .row:not(.shown) .hyp, body.blind .row:not(.shown) .name small { display: none; }
.v .name small { display: block; }
body:not(.blind) .reveal { display: none; }
.foot { margin-top: 16px; color: var(--ink-2); font-size: 12px; }
@media (max-width: 640px) {
  .picker { grid-template-columns: 1fr; }
  .row { grid-template-columns: 1fr; }
  .vers { grid-template-columns: 1fr; }
  .v { display: grid; grid-template-columns: 118px minmax(0, 1fr); column-gap: 8px; align-items: start; }
  .v .name { margin: 2px 0 0; }
  .v .stats, .v .hyp { grid-column: 2; }
}
"""

SCRIPT = r"""
const DATA = __DATA__;
const AUDIO = __AUDIO__;
const CONDS = __CONDS__;   // [{key, group, title, prec, steps, decode, wm, speed}]
const CURRENT = __CURRENT__;
const DEFAULT = __DEFAULT__;
const BY = Object.fromEntries(CONDS.map(c => [c.key, c]));
const PLAY = '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>';
const STOP = '<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1"/></svg>';
const list = document.getElementById('list');
const sortSel = document.getElementById('sort');
const seedSel = document.getElementById('seed');
const blind = document.getElementById('blind');
const count = document.getElementById('count');
const picker = document.getElementById('picker');
let current = null;  // {audio, btn}

for (const c of CONDS) {
  const lab = document.createElement('label');
  const fixed = c.key === CURRENT;
  if (fixed) lab.className = 'fixed';
  lab.innerHTML = '<input type="checkbox" value="' + c.key + '"' + (fixed || DEFAULT.includes(c.key) ? ' checked' : '') + (fixed ? ' disabled' : '') + '> ' + c.title;
  picker.appendChild(lab);
}
function selected() { return [...picker.querySelectorAll('input:checked')].map(x => x.value); }
function stop() {
  if (!current) return;
  current.audio.pause(); current.audio.currentTime = 0;
  current.btn.classList.remove('on'); current.btn.querySelector('.ic').innerHTML = PLAY;
  current.btn.querySelector('.fill').style.width = '0';
  current = null;
}
function play(btn, key) {
  const same = current && current.btn === btn;
  stop();
  if (same) return;
  const audio = new Audio('data:audio/mpeg;base64,' + AUDIO[key]);
  current = { audio, btn };
  btn.classList.add('on'); btn.querySelector('.ic').innerHTML = STOP;
  audio.addEventListener('timeupdate', () => {
    if (audio.duration) btn.querySelector('.fill').style.width = (100 * audio.currentTime / audio.duration) + '%';
  });
  audio.addEventListener('ended', () => { if (current && current.audio === audio) stop(); });
  audio.play();
}
// 名前を隠すときの並び。文とシードから決まる（開き直しても同じ）。
function perm(row, n) {
  let h = row.i * 7919 + row.seed * 104729 + 17;
  const idx = [...Array(n).keys()];
  for (let k = n - 1; k > 0; k--) { h = (h * 1103515245 + 12345) % 2147483648; const j = h % (k + 1); [idx[k], idx[j]] = [idx[j], idx[k]]; }
  return idx;
}
function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function signed(x) { return (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3); }
function render() {
  stop();
  const isBlind = blind.checked;
  document.body.classList.toggle('blind', isBlind);
  const keys = selected();
  const others = keys.filter(k => k !== CURRENT && BY[k].group !== 'prod');
  // 選んだ版のうち、今のエンジンより声の近さがいちばん下がったものの差
  const drop = r => others.length ? Math.min(...others.map(k => r.v[k].sim - r.v[CURRENT].sim)) : 0;
  let rows = DATA.filter(r => seedSel.value === 'all' || String(r.seed) === seedSel.value);
  rows = rows.slice().sort(sortSel.value === 'drop' ? (a, b) => drop(a) - drop(b) || a.i - b.i || a.seed - b.seed
                                                    : (a, b) => a.i - b.i || a.seed - b.seed);
  count.textContent = rows.length + ' 組 × ' + keys.length + ' 版';
  list.innerHTML = '';
  const LET = 'ABCDEFGHIJKLMNOPQ';
  for (const r of rows) {
    const el = document.createElement('div');
    el.className = 'row';
    const order = isBlind ? perm(r, keys.length) : keys.map((_, k) => k);
    const full = k => BY[k].title + '（' + BY[k].prec + '・' + BY[k].steps + ' ステップ）';
    const vers = order.map((vi, pos) => {
      const k = keys[vi], v = r.v[k], c = BY[k];
      const sub = '<small>' + c.prec + '・' + c.steps + ' ステップ・復号 ' + c.decode + '</small>' +
        '<small>冷えた状態の中央値 ' + esc(c.speed) + '</small>';
      const label = isBlind ? '<span class="blindname">' + LET[pos] + '</span>' + sub : c.title + sub;
      const errCls = v.err > 0 ? ' class="bad"' : '';
      const hyp = v.err > 0 ? '<div class="hyp">書き起こし「' + esc(v.hyp) + '」</div>' : '';
      const d = v.sim - r.v[CURRENT].sim;
      const delta = k === CURRENT ? '' : '（今のエンジンとの差 <span class="delta' + (d < -0.05 ? ' bad' : '') + '">' + signed(d) + '</span>）';
      return '<div class="v' + (k === CURRENT && !isBlind ? ' cur' : '') + '"><div class="name">' + label + '</div>' +
        '<button class="play" data-key="' + v.file + '"><span class="fill"></span><span class="ic">' + PLAY + '</span>再生<span class="t">' + v.sec.toFixed(1) + ' 秒</span></button>' +
        '<div class="stats"><span>声の近さ ' + v.sim.toFixed(3) + '</span>' + delta + '・<span' + errCls + '>読み誤り ' + v.err + '/' + v.n + ' 字</span>' +
        (v.wm === false ? '・<span class="bad">透かしを読み切れず</span>' : '') + '</div>' + hyp + '</div>';
    }).join('');
    el.innerHTML = '<div><div class="text">' + esc(r.text) + '</div><div class="meta">文 ' + r.i + '・シード ' + r.seed +
      (others.length ? '・今のエンジンとの声の近さの差 ' + signed(drop(r)) + '（選んだ版の中でいちばん下がったもの）' : '') + '</div>' +
      '<button class="reveal">答えを見る</button></div><div class="vers">' + vers + '</div>';
    el.querySelector('.meta').style.display = isBlind ? 'none' : '';
    el.querySelector('.reveal').addEventListener('click', ev => {
      el.classList.add('shown');
      el.querySelectorAll('.blindname').forEach((s, pos) => { s.textContent = LET[pos] + ' = ' + full(keys[order[pos]]); });
      el.querySelector('.meta').style.display = '';
      ev.target.remove();
    });
    el.querySelectorAll('.play').forEach(b => b.addEventListener('click', () => play(b, b.dataset.key)));
    list.appendChild(el);
  }
}
if (location.hash === '#blind') blind.checked = true;
[sortSel, seedSel, blind].forEach(x => x.addEventListener('change', render));
picker.addEventListener('change', render);
render();
"""


def summary_table(conds: list[dict], stats: dict) -> str:
    head = ("<tr><th>版</th><th>精度</th><th>ステップ</th><th>復号</th><th>透かし</th><th>冷えた状態の中央値</th><th>CER</th>"
            "<th>誤りのある音声</th><th>声の近さ（平均）</th><th>最小</th><th>今のエンジンとの差（平均）</th><th>いちばん下がった音声</th>"
            "<th>0.05 以上下がった音声</th></tr>")
    body, prev = [], None
    for c in conds:
        st = stats[c["key"]]
        cls = ("cur" if c["key"] == CURRENT else "") + (" sep" if prev is not None and prev != c["group"] and c["group"] == "old" else "")
        prev = c["group"]
        cur = c["key"] == CURRENT
        body.append(
            f'<tr class="{cls.strip()}"><td>{html.escape(c["title"])}<small>{html.escape(c["what"])}</small></td>'
            f'<td>{c["prec"]}</td><td>{c["steps"]}</td><td>{c["decode"]}</td><td>{c["wm"]}</td><td>{html.escape(c["speed"])}</td>'
            f'<td>{st["cer"]:.2f}%</td><td>{st["bad"]}/{st["n"]}</td><td>{st["sim"]:.3f}</td><td>{st["sim_min"]:.3f}</td>'
            + ("<td>—</td><td>—</td><td>—</td>" if cur else
               f'<td>{st["d_mean"]:+.3f}</td><td class="{"bad" if st["d_min"] < -0.05 else ""}">{st["d_min"]:+.3f}</td>'
               f'<td class="{"bad" if st["d_05"] else ""}">{st["d_05"]}/{st["n"]}</td>') + "</tr>")
    return f'<div class="tablewrap"><table>{head}{"".join(body)}</table></div>'


def build(out: Path, conds: list[dict], rows: list[dict], stats: dict, bitrate: str, *, default: list[str], scope: str,
          lead: str, sort_default: str = "drop") -> None:
    keys = [c["key"] for c in conds]
    rows = [{**r, "v": {k: r["v"][k] for k in keys}} for r in rows]
    audio = {v["file"]: mp3_b64(Q / v["file"], bitrate) for r in rows for v in r["v"].values()}
    script = (SCRIPT.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace("__AUDIO__", json.dumps(audio))
              .replace("__CONDS__", json.dumps(conds, ensure_ascii=False)).replace("__CURRENT__", json.dumps(CURRENT))
              .replace("__DEFAULT__", json.dumps(default)))
    opts = [("drop", "今のエンジンより声の近さが下がった順"), ("index", "文の番号順")]
    opt_html = "".join(f'<option value="{v}"{" selected" if v == sort_default else ""}>{t}</option>' for v, t in opts)
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>聞き比べ</title><style>{STYLE}</style></head>
<body><div class="page">
<h1>聞き比べ: 今のエンジンと、出力が変わる版</h1>
<p class="lead">{lead}</p>
{summary_table(conds, stats)}
<fieldset class="picker" id="picker"><legend>比べる版（今のエンジンはいつも出ます）</legend></fieldset>
<div class="bar">
  <label>並べ方 <select id="sort">{opt_html}</select></label>
  <label>シード <select id="seed"><option value="all">1 と 2</option><option value="1">1</option><option value="2">2</option></select></label>
  <label><input type="checkbox" id="blind"> 名前を隠す</label>
  <span class="count" id="count"></span>
</div>
<div id="list"></div>
<p class="foot">{scope}。表の数値は、どのページでも 48 文 × シード 1, 2（96 本）の集計です。
声の近さは microsoft/wavlm-base-plus-sv で測った、参照した話者の本物の 2 区間との近さです（本物どうしは 0.952）。
読み誤りは faster-whisper（medium）の書き起こしを、ひらがなに直して比べた字数です。「三回」と「3回」のような表記の違いも数えています。
冷えた状態の中央値は、起動の準備だけを済ませた新しいプロセスで、48 文 × シード 1, 2 を透かしなしで測った 1 文の合成時間です（2 回の中央値）。
「並べ方」の差は、選んだ版（今の本番を除く）の中で、今のエンジンよりいちばん下がったものの差です。</p>
</div>
<script>{script}</script>
</body></html>
"""
    out.write_text(page, encoding="utf-8")
    print(out, f"{len(rows)} 組 × {len(conds)} 版", f"{len(page) / 1e6:.2f} MB")


LEAD = ("同じ文を、同じノイズ（同じシード）から合成しました。「今のエンジン」は、本番に入れる候補の構成（fused、BF16、40 ステップ、"
        "復号を決まった長さで動かす、透かしなし）です。今回の版は、どれも今のエンジンと同じ条件（透かしなし、復号を決まった長さで動かす）で合成し、"
        "その版が変える部分だけを変えています。「今の本番」と「公式の 20 ステップ」は公式の実装（fp32、透かしあり）です。"
        "「名前を隠す」にすると、文ごとに並びが入れ替わり、A、B、C… の名前だけが出ます。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--bitrate", default="40k")
    ap.add_argument("--limit", type=int, default=0, help="--focus の版が今のエンジンより声の近さを下げた順に、この数だけ入れる")
    ap.add_argument("--focus", default="", help="今の本番・今のエンジン・この版だけのページを作る")
    ap.add_argument("--tops", type=int, default=0, help="今回の版ごとに、声の近さが下がった順のページ listen-top<数>-<条件>.html を作る")
    ap.add_argument("--no-old", action="store_true", help="前に聞いた版（透かしあり）を入れない")
    args = ap.parse_args()
    conds = [dict(c) for c in CONDS if (Q / f"rows-{c['key']}.json").exists() and not (args.no_old and c["group"] == "old")]
    missing = [c["key"] for c in CONDS if not (Q / f"rows-{c['key']}.json").exists()]
    if missing:
        print("音声がないので入れない条件:", missing)
    for c in conds:
        c["speed"] = cold_median(c["key"]) or c.get("speed") or "—"
    rows, stats = load(conds)

    def focus_page(key: str, limit: int, out: Path) -> None:
        sub = [c for c in conds if c["key"] in (PROD, CURRENT, key)]
        title = next(c["title"] for c in conds if c["key"] == key)
        rs = sorted(rows, key=lambda r: (r["v"][key]["sim"] - r["v"][CURRENT]["sim"], r["i"], r["seed"]))[: limit or len(rows)]
        build(out, sub, rs, stats, args.bitrate, default=[PROD, key],
              scope=f"「{title}」が今のエンジンより声の近さを下げた順の {len(rs)} 組",
              lead=LEAD + f"このページは、「{title}」の声の近さが今のエンジンよりいちばん下がった {len(rs)} 組です。")

    if args.tops:
        for c in conds:
            if c["group"] == "new":
                focus_page(c["key"], args.tops, Q / f"listen-top{args.tops}-{c['key']}.html")
        return
    if args.focus:
        focus_page(args.focus, args.limit, Path(args.out or Q / f"listen-{args.focus}.html"))
        return
    default = [PROD] + [k for k in (f"{E}-bdec-s20", f"{E}-bdec-fp8") if any(c["key"] == k for c in conds)]
    build(Path(args.out or Q / "listen.html"), conds, rows, stats, args.bitrate, default=default, scope="48 文 × シード 1, 2", lead=LEAD)


if __name__ == "__main__":
    main()
