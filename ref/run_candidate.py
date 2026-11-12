"""候補のエンジンを参照（runs/ref/<tag>/）と同じ入力で動かし、比べられる形で runs/cand/<name>/ に書く。

ここでの候補は公式の実装そのもの（--device cuda / cpu）。用途は 2 つ。
- --device cuda: 比べる仕組みの自己確認。参照と同じ計算なので、ほぼすべての段階で誤差 0 になるはず。
- --device cpu : 公式どうしの差（GPU と CPU）。移植した側の SNR を読むときの下限の目安にする。
移植したエンジンも、同じ形のファイルを書けば tools/snr.py で比べられる（形は docs/reference-format.md）。

書くもの（ケースごと）:
- ステップごとに参照の入力を入れた結果（teacher forcing）
  cond.npz（条件のエンコード）、kv.npz（参照の条件から作った KV）、steps_tf.npz（参照の x から計算した v）、
  blocks_s<k>.npz（参照の x から計算した各層の出力）、audio_tf.npz（参照の z の復号、参照の音声への電子透かし）
- 最初のノイズから通した結果: free.npz（各ステップの x、最後の x、復号、電子透かし、24kHz の出力）

使い方: scripts/run_in_image.sh ref/run_candidate.py --device cpu --name official-cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
from common import OUTPUT_SR, SPEAKER_SCALE, TEXT_SCALE, WORK, load_runtime, steps_tag  # noqa: E402
from dump_reference import BLOCK_TRACE_STEPS, KV_LAYERS, np32, stack  # noqa: E402
from irodori_tts.inference_runtime import find_flattening_point  # noqa: E402
from irodori_tts.meanflow import sample_euler_meanflow  # noqa: E402
from irodori_tts.rf import sample_euler_rf_cfg  # noqa: E402

sys.path.insert(0, "/app")
from trim import trim_audio  # noqa: E402
import torchaudio  # noqa: E402


def load(p: Path) -> dict[str, np.ndarray]:
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


class OfficialCandidate:
    def __init__(self, device: str):
        self.rt = load_runtime(device)
        self.model = self.rt.model
        self.meanflow = str(self.model.cfg.flow_parameterization) == "meanflow"
        self.device = torch.device(device)
        self.trace: dict[str, np.ndarray] | None = None
        for li, block in enumerate(self.model.blocks):
            block.register_forward_hook(self._hook(f"block{li:02d}_out"))
        b0 = self.model.blocks[0]
        for name, mod in (("b0_attention_adaln", b0.attention_adaln), ("b0_attention", b0.attention),
                          ("b0_mlp_adaln", b0.mlp_adaln), ("b0_mlp", b0.mlp)):
            mod.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def hook(_m, _i, out):
            if self.trace is None:
                return
            if isinstance(out, tuple):
                self.trace[name + "_h"] = np32(out[0])
                self.trace[name + "_gate"] = np32(out[1])
            else:
                self.trace[name] = np32(out)
        return hook

    def T(self, a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a).to(self.device)
        return t if t.dtype in (torch.bool, torch.int64) else t.to(self.model.dtype)

    @torch.inference_mode()
    def encode(self, inp: dict) -> tuple:
        return self.model.encode_conditions(
            text_input_ids=self.T(inp["text_ids"]), text_mask=self.T(inp["text_mask"]),
            ref_latent=self.T(inp["ref_latent"]), ref_mask=self.T(inp["ref_mask"]),
            caption_input_ids=self.T(inp["caption_ids"]), caption_mask=self.T(inp["caption_mask"]),
            speaker_uncond_mode="mask")

    @staticmethod
    def cfg_batch(c: dict) -> dict:
        """公式の independent の並び（cond、文を外したもの、話者を外したもの）。"""
        z = torch.zeros_like
        return {
            "text_state": torch.cat([c["text_state"], z(c["text_state"]), c["text_state"]]),
            "text_mask": torch.cat([c["text_mask"], z(c["text_mask"]), c["text_mask"]]),
            "speaker_state": torch.cat([c["speaker_state"], c["speaker_state"], z(c["speaker_state"])]),
            "speaker_mask": torch.cat([c["speaker_mask"], c["speaker_mask"], z(c["speaker_mask"])]),
            "caption_state": torch.cat([c["caption_state"]] * 3),
            "caption_mask": torch.cat([c["caption_mask"]] * 3),
        }

    @torch.inference_mode()
    def teacher_forced(self, ref_dir: Path, out_dir: Path) -> None:
        inp, rcond, rsteps = load(ref_dir / "inputs.npz"), load(ref_dir / "cond.npz"), load(ref_dir / "steps.npz")
        names = ("text_state", "text_mask", "speaker_state", "speaker_mask", "caption_state", "caption_mask")
        own = self.encode(inp)
        np.savez(out_dir / "cond.npz", **{k: np32(v) for k, v in zip(names, own)})
        c1 = {k: self.T(rcond[k]) for k in names}
        c3 = self.cfg_batch(c1)
        kv1 = self.model.build_context_kv_cache(text_state=c1["text_state"], speaker_state=c1["speaker_state"],
                                                caption_state=c1["caption_state"])
        kv3 = self.model.build_context_kv_cache(text_state=c3["text_state"], speaker_state=c3["speaker_state"],
                                                caption_state=c3["caption_state"])
        kv_out = {}
        for li in KV_LAYERS:
            for j, t in enumerate(kv1[li]):
                kv_out[f"kv_l{li % len(kv1)}_{j}"] = np32(t)
        np.savez(out_dir / "kv.npz", **kv_out)
        cfg_steps = set(int(i) for i in rsteps["cfg_steps"])
        v_cfg, v_nocfg, v_all = [], [], []
        for i in range(len(rsteps["t"])):
            x = self.T(rsteps["x"][i])
            t = torch.tensor([float(rsteps["t"][i])], device=self.device, dtype=self.model.dtype)
            self.trace = {} if i in BLOCK_TRACE_STEPS else None
            if i in cfg_steps:
                raw = self.model.forward_with_encoded_conditions(
                    x_t=torch.cat([x] * 3), t=t.repeat(3), latent_mask=None, context_kv_cache=kv3, **c3)
                c = raw.chunk(3, dim=0)
                v = c[0]
                for scale, chunk in zip((TEXT_SCALE, SPEAKER_SCALE), c[1:]):
                    v = v + scale * (c[0] - chunk)
                v_cfg.append(np32(raw))
            else:
                mf = {}
                if self.meanflow:  # 公式の sample_euler_meanflow と同じ delta（fp32 の t − t_next）
                    ts = inp["t_schedule"]
                    mf["delta_t"] = torch.tensor([float(ts[i] - ts[i + 1])], device=self.device, dtype=torch.float32)
                    t = t.to(torch.float32)
                raw = self.model.forward_with_encoded_conditions(
                    x_t=x, t=t, latent_mask=None, context_kv_cache=kv1, **c1, **mf)
                v = raw
                v_nocfg.append(np32(raw))
            v_all.append(np32(v))
            if self.trace is not None:
                np.savez(out_dir / f"blocks_s{i:02d}.npz", **self.trace)
            self.trace = None
        np.savez(out_dir / "steps_tf.npz", v_raw_cfg=stack(v_cfg), v_raw_nocfg=stack(v_nocfg), v=np.stack(v_all))
        ra = load(ref_dir / "audio.npz")
        dec = self.rt.codec.decode_latent(self.T(ra["decode_in"]))
        wm = self.rt.watermarker.encode_batch([torch.from_numpy(ra["wm_in"])], sample_rate=self.rt.codec.sample_rate)
        np.savez(out_dir / "audio_tf.npz", decode_out=np32(dec), wm_out=np32(wm[0]))

    @torch.inference_mode()
    def free_run(self, ref_dir: Path, out_dir: Path, meta: dict) -> None:
        """公式の synthesize と同じ順で、参照の入力（文字の変換、参照音声の潜在変数、ノイズ）から最後まで通す。"""
        inp = load(ref_dir / "inputs.npz")
        xs = []
        orig = self.model.forward_with_encoded_conditions

        def rec(**kw):
            xs.append(np32(kw["x_t"][:1]))
            return orig(**kw)

        self.model.forward_with_encoded_conditions = rec
        try:
            z = sample_euler_meanflow(
                model=self.model,
                text_input_ids=self.T(inp["text_ids"]), text_mask=self.T(inp["text_mask"]),
                ref_latent=self.T(inp["ref_latent"]), ref_mask=self.T(inp["ref_mask"]),
                sequence_length=int(inp["noise"].shape[1]),
                caption_input_ids=self.T(inp["caption_ids"]), caption_mask=self.T(inp["caption_mask"]),
                speaker_uncond_mode="mask", num_steps=meta["num_steps"], initial_noise=self.T(inp["noise"]),
            ) if self.meanflow else sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=self.T(inp["text_ids"]), text_mask=self.T(inp["text_mask"]),
                ref_latent=self.T(inp["ref_latent"]), ref_mask=self.T(inp["ref_mask"]),
                sequence_length=int(inp["noise"].shape[1]),
                caption_input_ids=self.T(inp["caption_ids"]), caption_mask=self.T(inp["caption_mask"]),
                speaker_uncond_mode="mask", num_steps=meta["num_steps"],
                cfg_scale_text=TEXT_SCALE, cfg_scale_caption=3.0, cfg_scale_speaker=SPEAKER_SCALE,
                cfg_guidance_mode="independent", cfg_min_t=0.5, cfg_max_t=1.0,
                use_context_kv_cache=True, t_schedule_mode="linear", sway_coeff=-1.0,
                initial_noise=self.T(inp["noise"]),
            )
        finally:
            self.model.forward_with_encoded_conditions = orig
        # _finish_request と同じ: 長さで切り、復号し、話し終わりで切り、電子透かしを入れる。
        sr = int(self.rt.codec.sample_rate)
        hop = int(self.rt.codec.model.hop_length)
        target_samples = max(1, int(meta["seconds"] * sr))
        latent_steps = math.ceil(target_samples / hop)
        zz = z[:, :latent_steps]
        audio = self.rt.codec.decode_latent(zz[0:1]).cpu()[0]
        max_samples = target_samples
        fp = find_flattening_point(zz[0], window_size=20, std_threshold=0.05, mean_threshold=0.1)
        if fp * hop > 0:
            max_samples = min(max_samples, fp * hop)
        audio = audio[:, :max_samples]
        wm = self.rt.watermarker.encode_batch([audio], sample_rate=sr)[0]
        a = torch.as_tensor(wm, dtype=torch.float32).reshape(1, -1)
        if sr != OUTPUT_SR:
            a = torchaudio.functional.resample(a, sr, OUTPUT_SR)
        out24k = np.clip(trim_audio(a.reshape(-1).numpy(), OUTPUT_SR), -1.0, 1.0)
        np.savez(out_dir / "free.npz", x=np.stack(xs), x_final=np32(z), decode_out=np32(audio), wm_out=np32(wm),
                 out24k=out24k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default=None, help="runs/ref/ の下の参照。既定は gpu（MF 版は gpu-mf）")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    if args.ref is None:
        args.ref = "gpu" + steps_tag(None)
    torch.set_num_threads(args.threads)
    cand = OfficialCandidate(args.device)
    ref_root = WORK / "runs" / "ref" / args.ref
    out_root = WORK / "runs" / "cand" / args.name
    for ref_dir in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        meta = json.loads((ref_dir / "meta.json").read_text())
        out_dir = out_root / ref_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cand.teacher_forced(ref_dir, out_dir)
        cand.free_run(ref_dir, out_dir, meta)
        (out_dir / "meta.json").write_text(json.dumps({"candidate": f"official-{args.device}", **meta},
                                                      ensure_ascii=False, indent=1))
        print("done", ref_dir.name, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
