"""CUDA Graph の基準（ref/graphs_baseline.py）を、参照と同じ文とシードで通しで動かし、tools/snr.py で比べられる形で書く。

公式の runtime.synthesize の中の sample_euler_rf_cfg だけを GraphSampler に替える。ノイズは同じシードの CUDA の
乱数なので、参照と同じになる。各ステップの x はグラフの中なので書かない（最後の x、復号、電子透かし、出力を比べる）。

  scripts/run_in_image.sh ref/check_graphs.py --trim --name graphs-trim
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
import irodori_tts.inference_runtime as IR  # noqa: E402
import irodori_tts.model as M  # noqa: E402
from common import WORK, load_runtime, make_request, postprocess  # noqa: E402
from dump_reference import np32  # noqa: E402
from graphs_baseline import GraphSampler, _timestep_embedding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default="gpu")
    ap.add_argument("--tf32", action="store_true", help="行列積の TF32 を有効にする（公式の既定は無効）")
    args = ap.parse_args()
    rt = load_runtime()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    M.get_timestep_embedding = _timestep_embedding
    rt.model._rope_freqs(2048, rt.model.device)
    sampler = GraphSampler(rt.model, trim_padding=args.trim)
    got: dict = {}

    def sample(**kw):
        z = sampler.sample(**kw)
        got["z"] = np32(z)
        return z

    dec, wm = rt.codec.decode_latent, rt.watermarker.encode_batch

    def decode(z):
        a = dec(z)
        got["decode_out"] = np32(a)
        return a

    def watermark(audios, *, sample_rate):
        out = wm(audios, sample_rate=sample_rate)
        got["wm_out"] = np32(out[0])
        return out

    IR.sample_euler_rf_cfg = sample
    rt.codec.decode_latent = decode
    rt.watermarker.encode_batch = watermark
    ref_root = WORK / "runs" / "ref" / args.ref
    for ref_dir in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        meta = json.loads((ref_dir / "meta.json").read_text())
        req = make_request(meta["text"], meta["seconds"], meta["seed"])
        rt.synthesize(req, log_fn=None)  # グラフを記録する
        got.clear()
        out24k = postprocess(rt.synthesize(req, log_fn=None))
        out_dir = WORK / "runs" / "cand" / args.name / ref_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "free.npz", x_final=got["z"], decode_out=got["decode_out"], wm_out=got["wm_out"],
                 out24k=out24k)
        print("done", ref_dir.name, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
