"""公式の実装（GPU、fp32、本番と同じ設定）を動かし、段階ごとの入出力を runs/ref/<case>/ に保存する。

イメージの中のコードは書き換えず、関数を包んで値を取り出す。保存するもの（形は schema.md と同じ）:
- inputs.npz : 文字の変換の結果（text_ids/text_mask、caption_ids/caption_mask）、参照音声の潜在変数
               （ref_latent/ref_mask）、最初のノイズ（noise）、時刻の表（t_schedule）
- cond.npz   : 条件のエンコードの出力（text_state、text_mask、speaker_state、speaker_mask、caption_state、caption_mask）
- kv.npz     : 文と話者の KV の射影（cond の分。層 0 と最後の層）
- steps.npz  : 各ステップのモデルの入力 x（B=1 の分）と時刻 t、モデルの出力 v_raw（CFG のステップは 3 つ分）、
               CFG を合成した v、最後の x
- blocks_s<k>.npz : ステップ k の DiT の各層の出力と、層 0 の中間（AdaLN、attention、MLP）
- audio.npz  : 復号の入力 z、復号の出力（48kHz）、電子透かしの出力（48kHz）、24kHz に変換して切り落とした出力
- meta.json  : 文、シード、形、CFG の強さ、TF32 の設定、版

参照音声から作った値（ref_latent、speaker_state、音声）を含むので、runs/ は git に入れない。

使い方: scripts/run_in_image.sh ref/dump_reference.py [--cases 0,10,28,47] [--seed 1] [--steps 20]
  保存先の既定は runs/ref/gpu<steps_tag>（RF 版の 40 ステップは gpu、20 ステップは gpu-s20、MF 版の 4 ステップは gpu-mf）。
  MF 版は IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF で選ぶ。MF 版には CFG のステップがないので、
  v_raw_cfg は空（0 個）になる。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
import irodori_tts.inference_runtime as IR  # noqa: E402
from common import (  # noqa: E402
    CHECKPOINT, CHECKPOINT_REPO, MODEL_STEPS, SPEAKER_SCALE, TEXT_SCALE, WORK, load_runtime, load_sentences, make_request,
    normalize_text, postprocess, set_num_steps, steps_tag,
)

BLOCK_TRACE_STEPS = (0, 20)  # CFG ありの最初のステップと、CFG なしの最初のステップ
KV_LAYERS = (0, -1)


def np32(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    return t.numpy() if t.dtype == torch.bool or not t.is_floating_point() else t.float().numpy()


def stack(arrays: list[np.ndarray]) -> np.ndarray:
    """np.stack。空なら 0 個の配列（MF 版の v_raw_cfg）。"""
    return np.stack(arrays) if arrays else np.zeros((0,), dtype=np.float32)


def t_schedule_for(model, num_steps: int) -> torch.Tensor:
    """公式のサンプラーと同じ時刻の表（GPU で作る）。RF 版は (1 − linspace(0, 1)) × 0.999、MF 版は linspace(1, 0)。"""
    if str(model.cfg.flow_parameterization) == "meanflow":
        return torch.linspace(1.0, 0.0, num_steps + 1, device=model.device, dtype=torch.float32)
    u = torch.linspace(0.0, 1.0, num_steps + 1, device=model.device)
    return (1.0 - u) * 0.999


class Recorder:
    def __init__(self, rt):
        self.rt = rt
        self.model = rt.model
        self.reset()
        m = self.model
        self.sample_name = "sample_euler_meanflow" if str(m.cfg.flow_parameterization) == "meanflow" else "sample_euler_rf_cfg"
        self.orig = {
            "encode_conditions": m.encode_conditions,
            "build_context_kv_cache": m.build_context_kv_cache,
            "forward": m.forward_with_encoded_conditions,
            "sample": getattr(IR, self.sample_name),
            "decode": rt.codec.decode_latent,
            "watermark": rt.watermarker.encode_batch,
        }
        m.encode_conditions = self._encode
        m.build_context_kv_cache = self._kv
        m.forward_with_encoded_conditions = self._forward
        setattr(IR, self.sample_name, self._sample)
        rt.codec.decode_latent = self._decode
        rt.watermarker.encode_batch = self._watermark
        self.hooks = []
        for li, block in enumerate(m.blocks):
            self.hooks.append(block.register_forward_hook(self._block_hook(f"block{li:02d}_out")))
        b0 = m.blocks[0]
        for name, mod in (("b0_attention_adaln", b0.attention_adaln), ("b0_attention", b0.attention),
                          ("b0_mlp_adaln", b0.mlp_adaln), ("b0_mlp", b0.mlp)):
            self.hooks.append(mod.register_forward_hook(self._block_hook(name)))

    def reset(self):
        self.d: dict = {"steps_x": [], "steps_t": [], "v_raw": [], "kv_calls": 0}
        self.step = -1
        self.blocks: dict[int, dict[str, np.ndarray]] = {}

    def _encode(self, **kw):
        out = self.orig["encode_conditions"](**kw)
        for k in ("text_input_ids", "text_mask", "ref_latent", "ref_mask", "caption_input_ids", "caption_mask"):
            v = kw.get(k)
            if v is not None:
                self.d["in_" + k] = np32(v)
        names = ("text_state", "text_mask", "speaker_state", "speaker_mask", "caption_state", "caption_mask")
        for k, v in zip(names, out):
            if v is not None:
                self.d["cond_" + k] = np32(v)
        return out

    def _kv(self, **kw):
        out = self.orig["build_context_kv_cache"](**kw)
        if self.d["kv_calls"] == 0:  # 最初の呼び出しが cond の分（B=1）
            for li in KV_LAYERS:
                for j, t in enumerate(out[li]):
                    self.d[f"kv_l{li % len(out)}_{j}"] = np32(t)
        self.d["kv_calls"] += 1
        return out

    def _forward(self, **kw):
        self.step += 1
        x = kw["x_t"]
        self.d["steps_x"].append(np32(x[:1]))
        self.d["steps_t"].append(float(kw["t"][0]))
        out = self.orig["forward"](**kw)
        self.d["v_raw"].append(np32(out))
        return out

    def _block_hook(self, name):
        def hook(_mod, _inp, out):
            if self.step in BLOCK_TRACE_STEPS:
                if isinstance(out, tuple):  # LowRankAdaLN は (h, gate) を返す
                    self.blocks.setdefault(self.step, {})[name + "_h"] = np32(out[0])
                    self.blocks[self.step][name + "_gate"] = np32(out[1])
                else:
                    self.blocks.setdefault(self.step, {})[name] = np32(out)
        return hook

    def _sample(self, **kw):
        z = self.orig["sample"](**kw)
        self.d["sample_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(type(v)))
                                   for k, v in kw.items() if not isinstance(v, torch.Tensor) and k != "model"}
        self.d["z_patched"] = np32(z)
        return z

    def _decode(self, z):
        audio = self.orig["decode"](z)
        self.d.setdefault("decode_in", []).append(np32(z))
        self.d.setdefault("decode_out", []).append(np32(audio))
        return audio

    def _watermark(self, audios, *, sample_rate):
        out = self.orig["watermark"](audios, sample_rate=sample_rate)
        self.d["wm_in"] = np32(audios[0])
        self.d["wm_out"] = np32(out[0])
        self.d["wm_sr"] = int(sample_rate)
        return out


def save_case(rec: Recorder, out_dir: Path, meta: dict, out24k: np.ndarray, model) -> None:
    d = rec.d
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(d["steps_x"])
    t = np.asarray(d["steps_t"], dtype=np.float32)
    # 公式と同じ時刻の表（GPU で作る）。t_schedule[i] は steps_t[i] と一致するはず。
    t_schedule = t_schedule_for(model, num_steps).cpu().numpy()
    assert np.array_equal(t_schedule[:-1], t), "時刻の表が公式の記録と合わない"
    np.savez(out_dir / "inputs.npz",
             text_ids=d["in_text_input_ids"], text_mask=d["in_text_mask"],
             caption_ids=d["in_caption_input_ids"], caption_mask=d["in_caption_mask"],
             ref_latent=d["in_ref_latent"], ref_mask=d["in_ref_mask"],
             noise=d["steps_x"][0], t_schedule=t_schedule)
    np.savez(out_dir / "cond.npz", **{k[5:]: v for k, v in d.items() if k.startswith("cond_")})
    np.savez(out_dir / "kv.npz", **{k: v for k, v in d.items() if k.startswith("kv_l")})
    # CFG の合成を公式と同じ式・同じ順で計算し、次のステップの x と一致することを確かめる。
    v_comb, cfg_steps = [], []
    for i in range(num_steps):
        raw = torch.from_numpy(d["v_raw"][i])
        if raw.shape[0] == 3:
            c = raw.chunk(3, dim=0)
            v = c[0]
            for scale, chunk in zip((TEXT_SCALE, SPEAKER_SCALE), c[1:]):
                v = v + scale * (c[0] - chunk)
            cfg_steps.append(i)
        else:
            v = raw
        v_comb.append(v.numpy())
    xs = [torch.from_numpy(x) for x in d["steps_x"]] + [torch.from_numpy(d["z_patched"])]
    ts = torch.from_numpy(t_schedule)
    max_step_err = 0.0
    for i in range(num_steps):
        x_next = xs[i] + torch.from_numpy(v_comb[i]) * (ts[i + 1] - ts[i])
        max_step_err = max(max_step_err, float((x_next - xs[i + 1]).abs().max()))
    np.savez(out_dir / "steps.npz",
             x=np.stack(d["steps_x"]), t=t, x_final=d["z_patched"],
             v=np.stack(v_comb),
             v_raw_cfg=stack([d["v_raw"][i] for i in cfg_steps]),
             v_raw_nocfg=stack([d["v_raw"][i] for i in range(num_steps) if i not in cfg_steps]),
             cfg_steps=np.asarray(cfg_steps))
    for k, blk in rec.blocks.items():
        np.savez(out_dir / f"blocks_s{k:02d}.npz", **blk)
    np.savez(out_dir / "audio.npz",
             decode_in=d["decode_in"][0], decode_out=d["decode_out"][0],
             wm_in=d["wm_in"], wm_out=d["wm_out"], out24k=out24k)
    meta.update({
        "num_steps": num_steps, "cfg_steps": cfg_steps,
        "cfg_scales": {"text": TEXT_SCALE, "speaker": SPEAKER_SCALE},
        "cfg_combine_vs_next_x_maxabs": max_step_err,
        "kv_calls": d["kv_calls"], "wm_sr": d["wm_sr"],
        "shapes": {
            "text_ids": list(d["in_text_input_ids"].shape),
            "text_valid": int(d["in_text_mask"].sum()),
            "caption_ids": list(d["in_caption_input_ids"].shape),
            "caption_valid": int(d["in_caption_mask"].sum()),
            "ref_latent": list(d["in_ref_latent"].shape),
            "speaker_state": list(d["cond_speaker_state"].shape),
            "speaker_valid": int(d["cond_speaker_mask"].sum()),
            "caption_state": list(d["cond_caption_state"].shape),
            "latent": list(d["z_patched"].shape),
            "decode_out": list(d["decode_out"][0].shape),
            "out24k": list(out24k.shape),
        },
        "sample_kwargs": d["sample_kwargs"],
        "flags": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "torch": torch.__version__,
        "official_commit": "7ed4374",
        "checkpoint": CHECKPOINT_REPO,
        "flow_parameterization": str(model.cfg.flow_parameterization),
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0,10,28,47,5,33")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None, help="runs/ref/ の下の保存先。既定は gpu<steps_tag>（gpu、gpu-s20、gpu-mf など）")
    ap.add_argument("--steps", type=int, default=MODEL_STEPS,
                    help=f"サンプリングのステップ数（既定はモデルの既定の {MODEL_STEPS}）")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = "gpu" + steps_tag(args.steps)
    assert args.tag == "gpu" + steps_tag(args.steps) or args.tag not in ("gpu", "gpu-s20", "gpu-mf"), \
        "ステップ数かモデルの違う参照を、既定の名前に保存しようとした"
    assert CHECKPOINT["tag"].lstrip("-") in args.tag, f"このモデルの参照は、名前に {CHECKPOINT['tag'].lstrip('-')} を入れて保存する"
    set_num_steps(args.steps)
    rt = load_runtime()
    sentences = load_sentences()
    for text in ("準備ができました。", "今日もよろしくお願いします。"):  # 参照音声の潜在変数をキャッシュに入れる
        rt.synthesize(make_request(text, 2.0, 0), log_fn=None)
    rec = Recorder(rt)
    root = WORK / "runs" / "ref" / args.tag
    for i in (int(x) for x in args.cases.split(",")):
        s = sentences[i]
        text = normalize_text(s["text"])
        rec.reset()
        result = rt.synthesize(make_request(text, s["seconds"], args.seed), log_fn=None)
        out24k = postprocess(result)
        case = f"{i:02d}_s{args.seed}"
        meta = {"case": case, "index": i, "text": text, "seconds": s["seconds"], "seed": args.seed}
        save_case(rec, root / case, meta, out24k, rt.model)
        print(case, json.dumps(json.loads((root / case / "meta.json").read_text())["shapes"]),
              "cfg_combine_err", json.loads((root / case / "meta.json").read_text())["cfg_combine_vs_next_x_maxabs"],
              flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
