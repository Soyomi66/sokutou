"""Drama 3 の音声の声の近さを出す（microsoft/wavlm-base-plus-sv の埋め込みのコサイン類似度）。

参照を使った条件だけに使う（参照なしでは、文ごとにモデルが声を選ぶので、近さを比べる相手がない）。2 つの値を出す。
- 参照: 参照の WAV（--ref-wav）との近さ。
- 重心: 公式の実装の音声（--ref で選ぶ条件）の埋め込みの平均との近さ。自分自身を重心に入れないよう、シード 1 の音声は
  シード 2 の公式の音声の重心と、シード 2 の音声はシード 1 の重心と比べる（qwen3/score_sim.py と同じ）。
  公式の実装どうしの値（--ref の条件そのもの）が、この測り方での上限の目安になる。
イメージ（sokutou/drama3-official か sokutou/qwen3-dev）の中で GPU 0 で動かす:

  scripts/run_in_image.sh drama3/score_sim.py --ref drama3-official-ref --ref-wav runs/dev/ref0.wav \
      drama3-official-ref drama3-kernels-ref ...

入力: runs/quality-drama3/rows-*.json（drama3/quality_rows.py）。出力: runs/quality-drama3/sim-<ref>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMForXVector

REPO = Path(__file__).resolve().parents[1]
Q = REPO / "runs" / "quality-drama3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="重心を作る条件（公式の実装）")
    ap.add_argument("--ref-wav", default="", help="参照の WAV（リポジトリからの相対、または絶対のパス）")
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

    rows: dict[str, list[dict]] = {}
    for p in sorted(Q.glob("rows-*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.setdefault(r["cond"], []).append(r)
    embs = {c: {(r["seed"], r["i"]): emb(Q / r["file"]) for r in rows[c]} for c in dict.fromkeys([args.ref, *args.conds])}
    cent = {}
    for seed in (1, 2):
        v = torch.stack([e for (s, _), e in embs[args.ref].items() if s == seed])
        cent[seed] = torch.nn.functional.normalize(v.mean(0), dim=-1)
    ref = None
    if args.ref_wav:
        ref = emb(Path(args.ref_wav) if Path(args.ref_wav).is_absolute() else REPO / args.ref_wav)
    out = {"ref": args.ref, "ref_wav": args.ref_wav, "centroid": {}, "reference": {}}
    for c in args.conds:
        sims = {f"s{s}_{i:02d}": float(e @ cent[3 - s]) for (s, i), e in embs[c].items()}
        out["centroid"][c] = sims
        v = list(sims.values())
        line = f"{c}: n {len(v)} 重心 mean {sum(v) / len(v):.3f} min {min(v):.3f}"
        if ref is not None:
            rs = {f"s{s}_{i:02d}": float(e @ ref) for (s, i), e in embs[c].items()}
            out["reference"][c] = rs
            w = list(rs.values())
            line += f"  参照 mean {sum(w) / len(w):.3f} min {min(w):.3f}"
        print(line, flush=True)
    (Q / f"sim-{args.ref}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
