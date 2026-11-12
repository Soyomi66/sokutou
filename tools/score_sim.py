"""話者照合モデル（microsoft/wavlm-base-plus-sv）で、参照した話者の本物の 2 区間との近さ（コサイン類似度の平均）を出す。

作者の前の計測（非公開）の score_sim.py を、このリポジトリの置き場所（runs/quality/）に合わせて写したもの。
測り方は同じ。本物の 2 区間（runs/quality/voice/hold_a.wav、hold_b.wav。参照音声と重ならない区間）どうしの値を
最初に出し、前の計測（0.952〜0.953）と合うかを確かめる。イメージの中で GPU 0 で動かす:

  scripts/run_in_image.sh tools/score_sim.py [条件...]

出力: runs/quality/sim.json
"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMForXVector

Q = Path("/work/runs/quality")
dev = "cuda"
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().to(dev)


def load16(p: Path) -> torch.Tensor:
    a, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = torch.from_numpy(a.mean(axis=1))
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    return x


@torch.no_grad()
def emb(p: Path) -> torch.Tensor:
    x = fe(load16(p).numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
    return torch.nn.functional.normalize(model(**x).embeddings[0], dim=-1).cpu()


hold = [emb(Q / "voice" / "hold_a.wav"), emb(Q / "voice" / "hold_b.wav")]
anchor = float(hold[0] @ hold[1])
print("本物の 2 区間どうし:", round(anchor, 3), flush=True)
rows = [r for p in sorted(Q.glob("rows-*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
out = {"anchor": anchor, "sim": {}}
conds = sys.argv[1:]  # 条件を選ぶと、その条件だけを測り、ほかは前の結果を使う
prev = json.loads((Q / "sim.json").read_text(encoding="utf-8"))["sim"] if conds and (Q / "sim.json").exists() else {}
for r in rows:
    if conds and r["cond"] not in conds and r["file"] in prev:
        out["sim"][r["file"]] = prev[r["file"]]
        continue
    e = emb(Q / r["file"])
    out["sim"][r["file"]] = float(sum(float(e @ t) for t in hold) / len(hold))
(Q / "sim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for cond in dict.fromkeys(r["cond"] for r in rows):
    v = [out["sim"][r["file"]] for r in rows if r["cond"] == cond]
    print(cond, "n", len(v), "mean %.3f min %.3f" % (sum(v) / len(v), min(v)), flush=True)
print("ALL DONE", flush=True)
