"""復号の CUDA Graph（qwen3/fast.py の Qwen3Fast.decode）が、公式の speech_tokenizer.decode と同じ波形を出すかを確かめる。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/check_decode.py --model 0.6b

公式の実装の出力のコード（runs/bench/qwen3/official-<model>-wav/*.npy）を、両方で復号して SNR を出す。
公式どうしの差（同じコードを 2 回復号したもの）も出す（cuDNN の畳み込みの選び方で、同じ入力でも少し変わることがある）。
300 フレームを超える区切りの経路は、短いコードを 3 回つないだもの（約 2 倍から 3 倍の長さ）で確かめる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import load  # noqa: E402
from fast import Qwen3Fast  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def snr(ref: np.ndarray, x: np.ndarray) -> float:
    n = min(len(ref), len(x))
    e = float(np.sum((ref[:n] - x[:n]) ** 2))
    return float("inf") if e == 0 else 10 * np.log10(float(np.sum(ref[:n] ** 2)) / e)


def tail(ref: np.ndarray, x: np.ndarray) -> float:
    """差の 2 乗の和のうち、最後の 10% の区間にある割合（後ろを詰めた影響が前に漏れていれば、後ろに集まる）。"""
    n = min(len(ref), len(x))
    e = (ref[:n] - x[:n]) ** 2
    tot = float(e.sum())
    return 0.0 if tot == 0 else float(e[int(n * 0.9):].sum()) / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    tts = load(args.model, args.dtype, "sdpa")
    tok = tts.model.speech_tokenizer
    print("decoder dtype", next(tok.model.decoder.parameters()).dtype, "upsample", int(tok.model.decoder.total_upsample))
    eng = Qwen3Fast(tts)
    t0 = time.perf_counter()
    print("warm_decode", round(eng.warm_decode(), 2), "s")
    wav_dir = REPO / "runs" / "bench" / "qwen3" / f"official-{args.model}-wav"
    files = sorted(wav_dir.glob(f"official-{args.model}-1_s1_*.npy"))[: args.limit]
    rows = []
    long_codes = torch.cat([torch.from_numpy(np.load(f)).long() for f in files[:8]], dim=0)
    cases = [(f.stem, torch.from_numpy(np.load(f)).long().cuda()) for f in files] + [("long", long_codes.cuda())]
    for name, codes in cases:
        a, sr = tok.decode([{"audio_codes": codes}])
        b, _ = tok.decode([{"audio_codes": codes}])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        c, sr2 = eng.decode(codes)
        torch.cuda.synchronize()
        t_graph = time.perf_counter() - t1
        t1 = time.perf_counter()
        tok.decode([{"audio_codes": codes}])
        torch.cuda.synchronize()
        t_off = time.perf_counter() - t1
        a = np.asarray(a[0], dtype=np.float32)
        b = np.asarray(b[0], dtype=np.float32)
        row = {"name": name, "frames": int(codes.shape[0]), "len_equal": len(a) == len(c), "sr_equal": sr == sr2,
               "snr_official_twice": snr(a, b), "snr_graph": snr(a, c), "tail_err_frac": tail(a, c),
               "ms_official": 1000 * t_off,
               "ms_graph": 1000 * t_graph}
        rows.append(row)
        print(json.dumps(row), flush=True)
    s = [r["snr_graph"] for r in rows]
    print(f"SNR（グラフ）最小 {min(s):.1f} dB 中央値 {sorted(s)[len(s) // 2]:.1f} dB; 長さが同じ {all(r['len_equal'] for r in rows)}")
    print(f"時間の中央値 公式 {sorted(r['ms_official'] for r in rows)[len(rows) // 2]:.1f} ms、"
          f"グラフ {sorted(r['ms_graph'] for r in rows)[len(rows) // 2]:.1f} ms")


if __name__ == "__main__":
    main()
