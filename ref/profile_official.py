"""公式の実装のプロファイル。1 文の合成時間を再現し、1 ステップ約 19 ミリ秒の内訳を調べる。

A. 基準の再現: 48 文 × シード 1, 2 を本番と同じ設定で合成する（計測のための細工なし）。
B. 区間ごとの時間: 条件のエンコード、KV の準備、モデルを 1 回通す時間（一度に通す数ごと）を、
   前後で同期して測る。公式のループは毎ステップ t.item() で同期するので、同期を足しても時間はほぼ変わらない。
C. torch.profiler: 1 文について、モデルを 1 回通すごとのカーネル数と GPU の稼働時間、同期の回数を数える。
   プロファイラーは CPU 側の時間を水増しするので、壁時計の時間は B の値を使う。
D. GPU だけの時間: モデルを 1 回通す処理を CUDA Graph に記録して再生し、カーネルを起動する手間を除いた時間を測る。

出力: runs/profile/（baseline.jsonl、sections.jsonl、trace.json、summary.json）
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
from common import WORK, load_runtime, load_sentences, make_request, normalize_text, summarize, timed_synth  # noqa: E402

OUT = WORK / "runs" / "profile"
OUT.mkdir(parents=True, exist_ok=True)

print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
flags = {
    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "cudnn_version": torch.backends.cudnn.version(),
}
print("flags", flags, flush=True)

rt = load_runtime()
model = rt.model
sentences = load_sentences()
print("watermarker ready:", rt.watermarker.ready, "codec sr", rt.codec.sample_rate,
      "hop", int(rt.codec.model.hop_length), flush=True)

for text in ("準備ができました。", "今日もよろしくお願いします。"):
    timed_synth(rt, make_request(text, 2.0, 0))

# ---- A. 基準の再現 ---------------------------------------------------------------------------------
rows_a = []
with (OUT / "baseline.jsonl").open("w", encoding="utf-8") as f:
    for seed in (1, 2):
        for i, s in enumerate(sentences):
            _, r = timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], seed))
            r.update({"i": i, "seed": seed})
            rows_a.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("A total", summarize([r["total_sec"] for r in rows_a]), flush=True)
print("A sample_rf", summarize([r["stages"]["sample_rf"] for r in rows_a]), flush=True)

# ---- B. 区間ごとの時間 -------------------------------------------------------------------------------
sec: dict[str, list[float]] = defaultdict(list)
shapes: dict[str, int] = {}
orig_encode = model.encode_conditions
orig_kv = model.build_context_kv_cache
orig_fwd = model.forward_with_encoded_conditions


def _timed(name, fn):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        sec[name].append(time.perf_counter() - t0)
        return out
    return wrapper


def _fwd(*args, **kwargs):
    x_t = kwargs["x_t"]
    b = int(x_t.shape[0])
    shapes["latent_len"] = int(x_t.shape[1])
    shapes["text_len"] = int(kwargs["text_state"].shape[1])
    shapes["speaker_len"] = int(kwargs["speaker_state"].shape[1])
    return _timed(f"forward_b{b}", orig_fwd)(*args, **kwargs)


model.encode_conditions = _timed("encode_conditions", orig_encode)
model.build_context_kv_cache = _timed("build_context_kv_cache", orig_kv)
model.forward_with_encoded_conditions = _fwd

rows_b = []
with (OUT / "sections.jsonl").open("w", encoding="utf-8") as f:
    for i, s in enumerate(sentences):
        sec.clear()
        _, r = timed_synth(rt, make_request(normalize_text(s["text"]), s["seconds"], 1))
        row = {
            "i": i,
            "total_sec": r["total_sec"],
            "sample_rf": r["stages"]["sample_rf"],
            "decode_latent": r["stages"]["decode_latent"],
            "watermark": r["stages"].get("silentcipher_watermark", 0.0),
            "encode_conditions": sum(sec["encode_conditions"]),
            "build_context_kv_cache": sum(sec["build_context_kv_cache"]),
            "kv_calls": len(sec["build_context_kv_cache"]),
            "forward_b3_sum": sum(sec["forward_b3"]),
            "forward_b1_sum": sum(sec["forward_b1"]),
            "forward_b3_med": float(np.median(sec["forward_b3"])),
            "forward_b1_med": float(np.median(sec["forward_b1"])),
            "n_b3": len(sec["forward_b3"]),
            "n_b1": len(sec["forward_b1"]),
            **shapes,
        }
        row["loop_rest"] = row["sample_rf"] - row["encode_conditions"] - row["build_context_kv_cache"] \
            - row["forward_b3_sum"] - row["forward_b1_sum"]
        rows_b.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

keys_b = ["total_sec", "sample_rf", "encode_conditions", "build_context_kv_cache", "forward_b3_sum", "forward_b1_sum",
          "forward_b3_med", "forward_b1_med", "loop_rest", "decode_latent", "watermark", "latent_len", "text_len"]
summary_b = {k: summarize([r[k] for r in rows_b]) for k in keys_b}
for k, v in summary_b.items():
    print("B", k, {kk: round(vv, 4) for kk, vv in v.items()}, flush=True)

model.encode_conditions = orig_encode
model.build_context_kv_cache = orig_kv

# ---- C. torch.profiler（中央くらいの長さの 1 文）-----------------------------------------------------
lens = [r["latent_len"] for r in rows_b]
mid = int(np.argsort(lens)[len(lens) // 2])
mid_s = sentences[mid]
captured: dict[int, dict] = {}


def _fwd_annot(*args, **kwargs):
    b = int(kwargs["x_t"].shape[0])
    if b not in captured:  # D で使う入力を 1 組ずつ取っておく
        captured[b] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
                       if k != "context_kv_cache"}
        captured[b]["context_kv_cache"] = [tuple(t.clone() for t in layer) for layer in kwargs["context_kv_cache"]]
    with torch.profiler.record_function(f"fwd_b{b}"):
        return orig_fwd(*args, **kwargs)


model.forward_with_encoded_conditions = _fwd_annot
req = make_request(normalize_text(mid_s["text"]), mid_s["seconds"], 1)
timed_synth(rt, req)
acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
with torch.profiler.profile(activities=acts) as prof:
    with torch.profiler.record_function("synthesize"):
        timed_synth(rt, req)
trace_path = OUT / "trace.json"
prof.export_chrome_trace(str(trace_path))
model.forward_with_encoded_conditions = orig_fwd

trace = json.loads(trace_path.read_text())
events = trace["traceEvents"] if isinstance(trace, dict) else trace
kernels = [e for e in events if e.get("cat") == "kernel"]
gpu_annots = [e for e in events if e.get("cat") == "gpu_user_annotation" and str(e.get("name", "")).startswith("fwd_b")]
cpu_annots = [e for e in events if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("fwd_b")]
runtime_calls = Counter(e["name"] for e in events if e.get("cat") == "cuda_runtime")
per_fwd: dict[str, list[tuple[int, float]]] = defaultdict(list)
kernel_names_b: dict[str, Counter] = defaultdict(Counter)
kernel_time_b: dict[str, Counter] = defaultdict(Counter)
kernels.sort(key=lambda e: e["ts"])
for a in gpu_annots:
    t0, t1 = a["ts"], a["ts"] + a["dur"]
    ks = [k for k in kernels if t0 <= k["ts"] < t1]
    per_fwd[a["name"]].append((len(ks), sum(k["dur"] for k in ks) / 1000.0))
    for k in ks:
        kernel_names_b[a["name"]][k["name"][:90]] += 1
        kernel_time_b[a["name"]][k["name"][:90]] += k["dur"] / 1000.0
synth_ev = [e for e in events if e.get("name") == "synthesize" and e.get("cat") == "user_annotation"]
summary_c = {
    "sentence_index": mid,
    "text": mid_s["text"],
    "kernels_total": len(kernels),
    "kernel_ms_total": sum(k["dur"] for k in kernels) / 1000.0,
    "synth_wall_ms_under_profiler": synth_ev[0]["dur"] / 1000.0 if synth_ev else None,
    "cpu_fwd_ms_under_profiler": {n: float(np.median([e["dur"] / 1000.0 for e in cpu_annots if e["name"] == n]))
                                  for n in sorted({e["name"] for e in cpu_annots})},
    "per_forward": {n: {"calls": len(v), "kernels_med": float(np.median([c for c, _ in v])),
                        "gpu_kernel_ms_med": float(np.median([t for _, t in v]))} for n, v in per_fwd.items()},
    "top_kernels_by_time": {n: [(k, round(t, 3), kernel_names_b[n][k]) for k, t in c.most_common(12)]
                            for n, c in kernel_time_b.items()},
    "cuda_runtime_calls": dict(runtime_calls.most_common(15)),
}
print("C", json.dumps({k: v for k, v in summary_c.items() if k != "top_kernels_by_time"}, ensure_ascii=False,
                      indent=1), flush=True)

# ---- D. CUDA Graph で再生したときの、モデルを 1 回通す時間 -----------------------------------------------
summary_d = {}
for b, kw in sorted(captured.items()):
    static = dict(kw)
    with torch.inference_mode():
        for _ in range(3):
            orig_fwd(**static)
        torch.cuda.synchronize()
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            orig_fwd(**static)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / n * 1000
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                orig_fwd(**static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                out = orig_fwd(**static)
            g.replay()
            torch.cuda.synchronize()
            ref = orig_fwd(**static)
            diff = float((out - ref).abs().max())
            t0 = time.perf_counter()
            for _ in range(n):
                g.replay()
            torch.cuda.synchronize()
            graph_ms = (time.perf_counter() - t0) / n * 1000
            err = None
        except Exception as exc:  # 記録できなかったら理由を残す
            graph_ms, diff, err = None, None, repr(exc)
    summary_d[f"b{b}"] = {"eager_ms_no_sync_between": eager_ms, "graph_replay_ms": graph_ms,
                          "graph_vs_eager_maxabs": diff, "error": err,
                          "latent_len": int(kw["x_t"].shape[1]), "text_len": int(kw["text_state"].shape[1])}
    print("D", b, summary_d[f"b{b}"], flush=True)

summary = {
    "flags": flags,
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "A_total": summarize([r["total_sec"] for r in rows_a]),
    "A_sample_rf": summarize([r["stages"]["sample_rf"] for r in rows_a]),
    "A_stages_med": {k: float(np.median([r["stages"].get(k, 0.0) for r in rows_a])) for k in rows_a[0]["stages"]},
    "B": summary_b,
    "C": summary_c,
    "D": summary_d,
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("ALL DONE", flush=True)
