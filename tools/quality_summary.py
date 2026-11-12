"""runs/quality/ の採点（asr.json、sim.json、rows-*.json の電子透かし）を、条件ごとの表にする。

  python3 tools/quality_summary.py [--base official]
出力: 標準出力と runs/quality/summary.json。公式の実装（--base）と結果が違う音声の一覧も出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

Q = Path(__file__).resolve().parent.parent / "runs" / "quality"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="official")
    args = ap.parse_args()
    rows = [r for p in sorted(Q.glob("rows-*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
    asr = {r["file"]: r for r in json.loads((Q / "asr.json").read_text(encoding="utf-8"))} if (Q / "asr.json").exists() else {}
    small = {r["file"]: r for r in json.loads((Q / "asr-small.json").read_text(encoding="utf-8"))} \
        if (Q / "asr-small.json").exists() else {}
    simd = json.loads((Q / "sim.json").read_text(encoding="utf-8")) if (Q / "sim.json").exists() else {"sim": {}}
    sim = simd["sim"]
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    conds.sort(key=lambda c: (c != args.base, c))
    summary = {"anchor": simd.get("anchor"), "conds": {}}
    print("| 条件 | 音声 | CER | 誤りのある音声 | 5% 超 | 声の近さ（平均） | 声の近さ（最小） | 透かし 24kHz | 透かし 48kHz | 透かしを入れる前（対照） |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in conds:
        rs = [r for r in rows if r["cond"] == c]
        a = [asr[r["file"]] for r in rs if r["file"] in asr]
        s = [sim[r["file"]] for r in rs if r["file"] in sim]
        e, n = sum(x["err"] for x in a), sum(x["n"] for x in a)
        d = {
            "n": len(rs), "cer": e / n if n else None, "err_chars": e, "ref_chars": n,
            "bad": sum(x["err"] > 0 for x in a), "bad5": sum(x["err"] / x["n"] > 0.05 for x in a),
            "sim_mean": sum(s) / len(s) if s else None, "sim_min": min(s) if s else None,
        }
        if all(r.get("wm_24k") for r in rs):  # 透かしを外した条件（-nowm）には、透かしの記録がない
            d.update(wm_24k=sum(r["wm_24k"]["ok"] for r in rs), wm_48k=sum(r["wm_48k"]["ok"] for r in rs),
                     wm_before=sum(r["wm_48k_before"]["ok"] for r in rs),
                     wm_24k_conf_min=min((r["wm_24k"]["confidence"] or 0.0) for r in rs))
        summary["conds"][c] = d
        f = lambda v, fmt: "—" if v is None else fmt.format(v)  # noqa: E731
        print(f"| {c} | {d['n']} | {f(d['cer'] and 100 * d['cer'], '{:.2f}%') if d['cer'] is not None else '—'} | {d['bad']} | {d['bad5']} | "
              f"{f(d['sim_mean'], '{:.3f}')} | {f(d['sim_min'], '{:.3f}')} | "
              + (f"{d['wm_24k']}/{d['n']} | {d['wm_48k']}/{d['n']} | {d['wm_before']}/{d['n']} |" if "wm_24k" in d else "— | — | — |"))
    if summary["anchor"] is not None:
        print(f"\n本物の 2 区間どうしの近さ: {summary['anchor']:.3f}")
    # 公式と誤りの数が違う音声
    base = {(r["i"], r["seed"]): asr.get(r["file"]) for r in rows if r["cond"] == args.base}
    diffs = []
    for c in conds:
        for r in rows:
            if r["cond"] != c:
                continue
            x, b = asr.get(r["file"]), base.get((r["i"], r["seed"]))
            if x is None or (x["err"] == 0 and (c == args.base or b is None or b["err"] == 0)):
                continue
            sm = small.get(r["file"])
            diffs.append({"cond": c, "i": r["i"], "seed": r["seed"], "err": x["err"], "n": x["n"],
                          "base_err": None if b is None else b["err"], "text": r["text"], "hyp": x["hyp"],
                          "small_err": None if sm is None else sm["err"], "small_hyp": None if sm is None else sm["hyp"]})
    summary["errors"] = diffs
    print("\n誤りのあった音声（条件、文の番号、シード、誤りの字数 / 字数、公式の同じ文の誤り、small での誤り）:")
    for d in diffs:
        print(f"- {d['cond']} #{d['i']:02d} s{d['seed']}: {d['err']}/{d['n']}（公式 {d['base_err']}、small {d['small_err']}）"
              f" 文「{d['text']}」→ 書き起こし「{d['hyp']}」")
    (Q / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
