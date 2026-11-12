"""Qwen3-TTS の音声が、公式の実装の同じ話者の声にどれだけ近いかを出す（microsoft/wavlm-base-plus-sv）。

Qwen3-TTS の CustomVoice は決まった話者（Ono_Anna）で話すので、参照音声との近さ（tools/score_sim.py）は使えない。
代わりに、公式の実装の音声の埋め込みの平均（重心）との近さ（コサイン類似度）を測る。自分自身を重心に入れないよう、
シード 1 の音声はシード 2 の公式の音声の重心と、シード 2 の音声はシード 1 の公式の音声の重心と比べる。
公式の実装どうしの値（--ref で選んだ条件そのもの）が、この測り方での上限の目安になる。イメージの中で GPU 0 で動かす:

  scripts/run_in_image.sh qwen3/score_sim.py --ref qwen3-official-0.6b qwen3-official-0.6b qwen3-speechcpp-0.6b ...

入力: runs/quality-qwen3/rows-*.json（qwen3 の WAV の一覧）。出力: runs/quality-qwen3/sim-<ref>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMForXVector

Q = Path(__file__).resolve().parents[1] / "runs" / "quality-qwen3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="重心を作る条件（公式の実装）")
    ap.add_argument("conds", nargs="+")
    args = ap.parse_args()
    dev = "cuda"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().to(dev)

    @torch.no_grad()
    def emb(p: Path) -> torch.Tensor:
        a, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = torch.from_numpy(a.mean(axis=1))
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)
        f = fe(x.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
        return torch.nn.functional.normalize(model(**f).embeddings[0], dim=-1).cpu()

    rows = {}
    for p in sorted(Q.glob("rows-*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.setdefault(r["cond"], []).append(r)
    embs = {c: {(r["seed"], r["i"]): emb(Q / r["file"]) for r in rows[c]} for c in dict.fromkeys([args.ref, *args.conds])}
    cent = {}
    for seed in (1, 2):
        v = torch.stack([e for (s, _), e in embs[args.ref].items() if s == seed])
        cent[seed] = torch.nn.functional.normalize(v.mean(0), dim=-1)
    out = {"ref": args.ref, "sim": {}}
    for c in args.conds:
        sims = {f"s{s}_{i:02d}": float(e @ cent[3 - s]) for (s, i), e in embs[c].items()}
        out["sim"][c] = sims
        v = list(sims.values())
        print(f"{c}: n {len(v)} mean {sum(v) / len(v):.3f} min {min(v):.3f}", flush=True)
    (Q / f"sim-{args.ref}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
