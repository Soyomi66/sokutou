"""runs/quality/wav/ の音声を faster-whisper（medium）で書き起こし、ひらがなに直して文字誤り率（CER）を出す。

作者の前の計測（非公開）の score_asr.py を、このリポジトリの置き場所（runs/quality/）に合わせて写したもの。
書き起こしと正規化の方法は同じ（beam 5、language=ja、NFKC、記号を除く、pykakasi でひらがなに直す）。
GPU 0 で動かす。CUDA の番号は nvidia-smi と違うことがあるので、GPU 0 の UUID を CUDA_VISIBLE_DEVICES で渡す:

  CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) \\
    LD_LIBRARY_PATH=$(echo .venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':') \\
    .venv/bin/python tools/score_asr.py [条件...] [--small]   # 条件を選ぶと、その条件だけを書き起こして、前の結果に足す

出力: runs/quality/asr.json（書き起こしを含むので git に入れない）
QUALITY_DIR を指定すると、runs/quality/ の代わりにそのディレクトリの rows-*.json を読み、そこに書く
（Qwen3-TTS は runs/quality-qwen3/）。
"""

import json
import os
import re
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np
import pykakasi
from faster_whisper import WhisperModel

Q = Path(os.environ.get("QUALITY_DIR") or Path(__file__).resolve().parent.parent / "runs" / "quality").resolve()
rows = [r for p in sorted(Q.glob("rows-*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
kks = pykakasi.kakasi()


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\s、。！？!?,.・「」…ー〜~]", "", t)
    return "".join(x["hira"] for x in kks.convert(t))


def edit_distance(r: str, h: str) -> int:
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def load(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a, sr


def to16(a, sr):
    return np.interp(np.arange(0, len(a), sr / 16000), np.arange(len(a)), a).astype(np.float32)


model_name = "medium"
out_name = "asr.json"
conds = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--small" in sys.argv:  # 書き起こしの誤りと区別するための 2 つめの書き起こし
    model_name, out_name = "small", "asr-small.json"
model = WhisperModel(model_name, device="cuda", device_index=0, compute_type="float16")
out = []
for r in rows:
    if conds and r["cond"] not in conds:
        continue
    a, sr = load(Q / r["file"])
    parts, _ = model.transcribe(to16(a, sr), language="ja", beam_size=5)
    hyp = "".join(x.text for x in parts)
    ref = norm(r["text"])
    out.append({"cond": r["cond"], "i": r["i"], "seed": r["seed"], "file": r["file"], "text": r["text"],
                "hyp": hyp, "err": edit_distance(ref, norm(hyp)), "n": len(ref)})
if conds and (Q / out_name).exists():  # 条件を選んだときは、ほかの条件の前の結果を残す
    done = {r["file"] for r in out}
    out = [r for r in json.loads((Q / out_name).read_text(encoding="utf-8")) if r["file"] not in done] + out
(Q / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for cond in dict.fromkeys(r["cond"] for r in out):
    rs = [r for r in out if r["cond"] == cond]
    e = sum(r["err"] for r in rs)
    n = sum(r["n"] for r in rs)
    bad = sum(r["err"] > 0 for r in rs)
    bad5 = sum(r["err"] / r["n"] > 0.05 for r in rs)
    print(f"{cond}: CER {100 * e / n:.2f}%  誤りのある音声 {bad}/{len(rs)}  5%超 {bad5}", flush=True)
