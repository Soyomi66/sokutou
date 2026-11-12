"""文のエンコード（ModernBERT と text_norm）だけの時間を、公式の 256 トークンと、有効なトークンだけのときで比べる。

設計メモの見込みでは「有効なトークンだけ通すと約 0.01 秒」と仮定していた。その値を確かめる。
有効なトークンだけ通した結果が、256 トークンで通した結果の有効な行と同じかも SNR で見る。

公式の合成を途中まで動かし、`sample_euler_rf_cfg` に渡る引数（文のトークンとマスク）を取り出してから止める。
出力: runs/profile/text_encode.jsonl（1 文 1 行）と、標準出力の要約。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/work/ref")

import torch  # noqa: E402

from common import WORK, load_runtime, load_sentences, make_request  # noqa: E402

import irodori_tts.inference_runtime as IR  # noqa: E402

REPEAT = 7


class _Captured(Exception):
    pass


def encode_text(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if model.pretrained_text_backbone is None:
        state = model.text_encoder(ids, mask)
    else:
        state = model.text_encoder(model.pretrained_text_backbone, ids, mask)
    return model.text_norm(state)


def timed(fn) -> float:
    vals = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals[1:])  # 最初の 1 回は捨てる


def snr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = ref.double(), cand.double()
    err = float(((ref - cand) ** 2).sum())
    return math.inf if err == 0 else 10 * math.log10(float((ref ** 2).sum()) / err)


def main() -> None:
    rt = load_runtime("cuda")
    model = rt.model
    captured: dict = {}
    orig = IR.sample_euler_rf_cfg

    def grab(**kw):
        captured.update(kw)
        raise _Captured

    IR.sample_euler_rf_cfg = grab
    out = Path(WORK) / "runs/profile/text_encode.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for i, s in enumerate(load_sentences()):
            captured.clear()
            try:
                rt.synthesize(make_request(s["text"], s["seconds"], seed=1))
            except _Captured:
                pass
            ids, mask = captured["text_input_ids"], captured["text_mask"]
            n = int(mask.sum())
            assert bool(mask[:, :n].all()), "文のマスクが前詰めでない"
            with torch.inference_mode():
                full = encode_text(model, ids, mask)
                trim = encode_text(model, ids[:, :n], mask[:, :n])
                row = {
                    "i": i, "chars": len(s["text"]), "tokens": n, "padded": int(ids.shape[1]),
                    "full_sec": timed(lambda: encode_text(model, ids, mask)),
                    "trim_sec": timed(lambda: encode_text(model, ids[:, :n], mask[:, :n])),
                    "snr_db": snr_db(full[:, :n], trim),
                }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        IR.sample_euler_rf_cfg = orig
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for key in ("full_sec", "trim_sec", "snr_db"):
        vals = sorted(r[key] for r in rows)
        print(f"{key}: min {vals[0]:.4g} med {statistics.median(vals):.4g} max {vals[-1]:.4g}")


if __name__ == "__main__":
    main()
