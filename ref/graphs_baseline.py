"""基準: 公式のモジュールのまま、KV の準備と 40 ステップを 1 つの CUDA Graph にまとめて再生する。

公式の sample_euler_rf_cfg の代わりに、同じ計算を同期なしで並べた関数を runtime に差し込む。
- independent の CFG だけを扱う（本番の設定）。ほかの設定では公式の関数をそのまま呼ぶ。
- 毎ステップの t.item() と torch.full(..., t) をやめ、CFG をかけるステップは CPU で先に決める。
- get_timestep_embedding の torch.tensor(10000.0, device=...)（ホストからのコピー）は記録できないので、
  同じ値を一度だけ GPU に作って使い回す版に差し替える（計算は同じ）。
- グラフは (潜在の長さ, 文のトークン数, 話者のトークン数) ごとに記録する。長さを丸める工夫はしていない。
  48 文で出てくる長さは、計測の前の温めですべて記録しておき、記録にかかった時間は別に出す。

確かめること:
1. 同じシードで公式の実装と潜在変数・音声が一致するか（最大絶対誤差）。
2. 48 文 × シード 1, 2 の合成時間（timed_synth の範囲。公式の実装と同じ）。

出力: runs/graphs/（bench.jsonl、summary.json）
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
import irodori_tts.inference_runtime as IR  # noqa: E402
import irodori_tts.model as M  # noqa: E402
from common import WORK, load_runtime, load_sentences, make_request, normalize_text, summarize, timed_synth  # noqa: E402
from irodori_tts.rf import draw_initial_noise, sample_euler_rf_cfg  # noqa: E402

OUT = WORK / "runs" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

_LOG1E4: dict[torch.device, torch.Tensor] = {}


def _timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """M.get_timestep_embedding と同じ計算。log(10000) の GPU のテンソルを使い回す。"""
    assert dim % 2 == 0
    half = dim // 2
    dev = timestep.device
    if dev not in _LOG1E4:
        _LOG1E4[dev] = torch.log(torch.tensor(10000.0, device=dev, dtype=torch.float32))
    freqs = 1000.0 * torch.exp(
        -_LOG1E4[dev] * torch.arange(half, device=dev, dtype=torch.float32) / half
    )
    args = timestep[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)


class GraphSampler:
    def __init__(self, model, *, trim_padding: bool = False):
        self.model = model
        # True なら、パディングした文のトークンと空の caption を attention から外す（数学的には同じ計算）。
        self.trim_padding = trim_padding
        self.graphs: dict[tuple, dict] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_sec: list[float] = []

    def _loop(self, st: dict) -> None:
        """記録する本体。st の静的なテンソルだけを読み書きする。"""
        model = self.model
        kv_cond = model.build_context_kv_cache(
            text_state=st["text_c"], speaker_state=st["spk_c"], caption_state=st["cap_c"])
        kv_cfg = model.build_context_kv_cache(
            text_state=st["text_3"], speaker_state=st["spk_3"], caption_state=st["cap_3"])
        x = st["x"]
        ts = st["t_schedule"]
        for i in range(st["num_steps"]):
            t, t_next = ts[i], ts[i + 1]
            if st["use_cfg"][i]:
                x3 = torch.cat([x] * 3, dim=0)
                tt3 = t.reshape(1).expand(3).to(x.dtype)
                v_out = model.forward_with_encoded_conditions(
                    x_t=x3, t=tt3,
                    text_state=st["text_3"], text_mask=st["tmask_3"],
                    speaker_state=st["spk_3"], speaker_mask=st["smask_3"],
                    caption_state=st["cap_3"], caption_mask=st["cmask_3"],
                    latent_mask=None, context_kv_cache=kv_cfg)
                chunks = v_out.chunk(3, dim=0)
                v = chunks[0]
                for scale, chunk in zip(st["scales"], chunks[1:]):
                    v = v + scale * (chunks[0] - chunk)
            else:
                tt = t.reshape(1).to(x.dtype)
                v = model.forward_with_encoded_conditions(
                    x_t=x, t=tt,
                    text_state=st["text_c"], text_mask=st["tmask_c"],
                    speaker_state=st["spk_c"], speaker_mask=st["smask_c"],
                    caption_state=st["cap_c"], caption_mask=st["cmask_c"],
                    latent_mask=None, context_kv_cache=kv_cond)
            x = x + v * (t_next - t)
        st["out"] = x

    def sample(self, **kw) -> torch.Tensor:
        model = kw["model"]
        supported = (
            kw.get("cfg_guidance_mode") == "independent"
            and kw.get("initial_noise") is None and kw.get("latent_mask") is None
            and kw.get("truncation_factor") is None and kw.get("rescale_k") is None
            and kw.get("speaker_kv_scale") is None and kw.get("t_schedule_mode", "linear") == "linear"
            and kw.get("speaker_uncond_mode") == "mask" and kw.get("caption_input_ids") is not None
            and kw["text_input_ids"].shape[0] == 1 and kw.get("cfg_scale") is None
        )
        if not supported:
            return sample_euler_rf_cfg(**kw)
        device, dtype = model.device, model.dtype
        num_steps = int(kw.get("num_steps", 40))
        seq_len = int(kw["sequence_length"])
        x0, _, _ = draw_initial_noise(int(kw.get("seed", 0)), (1, seq_len, model.cfg.patched_latent_dim),
                                      device=device, dtype=dtype)
        text_c, tmask_c, spk_c, smask_c, cap_c, cmask_c = model.encode_conditions(
            text_input_ids=kw["text_input_ids"], text_mask=kw["text_mask"],
            ref_latent=kw["ref_latent"], ref_mask=kw["ref_mask"],
            caption_input_ids=kw["caption_input_ids"], caption_mask=kw["caption_mask"],
            speaker_state_override=kw.get("speaker_state_override"),
            speaker_mask_override=kw.get("speaker_mask_override"), speaker_uncond_mode="mask")
        # 本番の設定（文 3.0、話者 5.0、caption は空なので CFG なし）。公式の関数と同じ条件で確かめる。
        assert not bool(cmask_c.any()), "caption が空でないときは未対応"
        if self.trim_padding:
            n_text = int(tmask_c.sum())
            assert bool(tmask_c[:, :n_text].all()), "文のマスクが前詰めでない"
            text_c, tmask_c = text_c[:, :n_text], tmask_c[:, :n_text]
            cap_c, cmask_c = cap_c[:, :0], cmask_c[:, :0]
        scales = [float(kw["cfg_scale_text"]), float(kw["cfg_scale_speaker"])]
        assert all(s > 0 for s in scales)
        key = (seq_len, text_c.shape[1], spk_c.shape[1], cap_c.shape[1], num_steps,
               float(kw["cfg_min_t"]), float(kw["cfg_max_t"]), tuple(scales))
        g = self.graphs.get(key)
        inputs = {
            "x": x0, "text_c": text_c, "tmask_c": tmask_c, "spk_c": spk_c, "smask_c": smask_c,
            "cap_c": cap_c, "cmask_c": cmask_c,
            # 公式の independent の並び: cond、文を外したもの、話者を外したもの。
            "text_3": torch.cat([text_c, torch.zeros_like(text_c), text_c]),
            "tmask_3": torch.cat([tmask_c, torch.zeros_like(tmask_c), tmask_c]),
            "spk_3": torch.cat([spk_c, spk_c, torch.zeros_like(spk_c)]),
            "smask_3": torch.cat([smask_c, smask_c, torch.zeros_like(smask_c)]),
            "cap_3": torch.cat([cap_c] * 3), "cmask_3": torch.cat([cmask_c] * 3),
        }
        if g is None:
            t0 = time.perf_counter()
            u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
            t_schedule = (1.0 - u) * 0.999
            ts_cpu = t_schedule.cpu()
            use_cfg = [float(kw["cfg_min_t"]) <= float(ts_cpu[i]) <= float(kw["cfg_max_t"])
                       for i in range(num_steps)]
            st = {k: v.clone() for k, v in inputs.items()}
            st.update(t_schedule=t_schedule, num_steps=num_steps, use_cfg=use_cfg, scales=scales)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._loop(st)  # 温め（同じ入力で一度通す）
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._loop(st)
            torch.cuda.synchronize()
            g = {"graph": graph, "st": st}
            self.graphs[key] = g
            self.capture_sec.append(time.perf_counter() - t0)
        st = g["st"]
        for k, v in inputs.items():
            st[k].copy_(v)
        g["graph"].replay()
        return st["out"].clone()


def main() -> None:
    rt = load_runtime()
    sentences = load_sentences()
    M.get_timestep_embedding = _timestep_embedding
    # RoPE の表は、長い文が来ると作り直される。記録したグラフが古い表を指したままにならないよう、
    # 先に最大の長さ（30 秒ぶん）まで作っておく。短い位置の値は作り直しても同じ。
    rt.model._rope_freqs(2048, rt.model.device)
    official = IR.sample_euler_rf_cfg
    sampler = GraphSampler(rt.model)

    # 潜在変数を取り出すために、runtime が呼ぶ関数を包む。
    last: dict[str, torch.Tensor] = {}

    def use(fn):
        def wrapper(**kw):
            z = fn(**kw)
            last["z"] = z.detach().clone()
            return z
        return wrapper

    # 1. 公式の実装との一致（同じシード）。差し替えた timestep embedding は公式の側にも効いている。
    checks = []
    for i in (0, 10, 28, 47):
        s = sentences[i]
        req = make_request(normalize_text(s["text"]), s["seconds"], 1)
        IR.sample_euler_rf_cfg = use(official)
        a_off, _ = timed_synth(rt, req)
        z_off = last["z"]
        IR.sample_euler_rf_cfg = use(sampler.sample)
        a_g, _ = timed_synth(rt, req)
        z_g = last["z"]
        n = min(len(a_off), len(a_g))
        err = a_off[:n] - a_g[:n]
        snr = 10 * math.log10(float(np.sum(a_off[:n] ** 2)) / max(float(np.sum(err ** 2)), 1e-30))
        row = {"i": i, "latent_maxabs": float((z_off - z_g).abs().max()), "latent_len": int(z_off.shape[1]),
               "audio_len_official": len(a_off), "audio_len_graph": len(a_g), "audio_snr_db": snr}
        checks.append(row)
        print("check", row, flush=True)

    # 2. 速さ。48 文の長さをすべて温めで記録してから測る。
    IR.sample_euler_rf_cfg = sampler.sample
    t0 = time.perf_counter()
    for s in sentences:
        timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], 0))
    warm_sec = time.perf_counter() - t0
    print("graphs", len(sampler.graphs), "capture med", np.median(sampler.capture_sec), "warm total", warm_sec,
          "mem GB", torch.cuda.memory_reserved() / 1e9, flush=True)
    rows = []
    with (OUT / "bench.jsonl").open("w", encoding="utf-8") as f:
        for seed in (1, 2):
            for i, s in enumerate(sentences):
                _, r = timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], seed))
                r.update({"i": i, "seed": seed})
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "checks": checks,
        "graphs": len(sampler.graphs),
        "capture_sec": summarize(sampler.capture_sec),
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total": summarize([r["total_sec"] for r in rows]),
        "sample_rf": summarize([r["stages"]["sample_rf"] for r in rows]),
        "stages_med": {k: float(np.median([r["stages"].get(k, 0.0) for r in rows])) for k in rows[0]["stages"]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=1), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
