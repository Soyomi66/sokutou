"""Qwen3-TTS（12Hz、CustomVoice）の公式の実装を、48 文 × シードで測る。イメージ sokutou/qwen3-official の中で動かす。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-official:0.1.1 scripts/run_in_image.sh \
    qwen3/bench.py --model 0.6b --name qwen3/official-0.6b-1

公式の generate_custom_voice（qwen-tts 0.1.1）をそのまま呼び、1 文の時間（合成から 24kHz の波形まで）と、
その内訳（言語モデルとコードの予測の generate、コーデックの復号）を記録する。話者は Ono_Anna、言語は Japanese。
--engine fast で qwen3/fast.py、--engine kernels で qwen3/kernels.py のエンジンを測る（内訳は prompt、generate、decode。
CUDA Graph の記録は測る前にすませる）。
生成の設定はモデルの generation_config.json の既定（サンプリングあり）。--greedy でいちばん確率の高いトークンを選ぶ。
シードは文ごとに torch.manual_seed で決める（公式の実装のサンプリングは torch の既定の乱数を使う）。
出力: runs/bench/<name>.jsonl（tools/bench_summary.py で集計できる形）、--save-dir で WAV とコード
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "85e237c12c027371202489a0ec509ded67b5e4b5"),
    "1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe"),
}
SPEAKER, LANGUAGE = "Ono_Anna", "Japanese"


def load(model: str, dtype: str, attn: str, device: str = "cuda:0") -> Qwen3TTSModel:
    from huggingface_hub import snapshot_download

    repo, rev = MODELS[model]
    path = snapshot_download(repo, revision=rev)
    return Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=getattr(torch, dtype), attn_implementation=attn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--dtype", default="bfloat16", help="公式の推奨は bfloat16")
    ap.add_argument("--attn", default="sdpa", help="flash_attention_2 は flash-attn が要る（イメージには入れていない）")
    ap.add_argument("--greedy", action="store_true", help="do_sample と subtalker_dosample を False にする")
    ap.add_argument("--save-dir", default="", help="runs/ の下のこのディレクトリに WAV（24kHz）とコード（.npy）を保存する")
    ap.add_argument("--limit", type=int, default=0, help="最初の N 文だけ（試すとき）")
    ap.add_argument("--engine", choices=["official", "fast", "kernels"], default="official")
    ap.add_argument("--wd", default="bf16", help="--engine kernels の重みの型（bf16、i8g32）")
    ap.add_argument("--sampler", choices=["fused", "torch"], default="fused",
                    help="--engine kernels のサンプリング（fused: 1 カーネル、torch: 公式と同じ乱数の torch の演算）")
    ap.add_argument("--l2-keep-mb", type=float, default=None,
                    help="--engine kernels で L2 に残す code predictor の重みの上限（MiB）。既定は L2 の大きさ - 16 MiB")
    ap.add_argument("--cp-pair", choices=["auto", "on", "off"], default="auto",
                    help="--engine kernels で code predictor の位置 0 と 1 をまとめて通すか（auto は on）")
    ap.add_argument("--max-cache", type=int, default=1024, help="--engine fast の言語モデルの KV キャッシュの長さ")
    ap.add_argument("--compile", action="store_true", help="--engine fast で 1 フレーム分を torch.compile する")
    args = ap.parse_args()

    tts = load(args.model, args.dtype, args.attn)
    if args.engine in ("fast", "kernels"):
        return bench_fast(tts, args)
    inner = tts.model
    stages: dict[str, float] = {}

    def timed(name, fn):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            stages[name] = time.perf_counter() - t0
            return out
        return wrapper

    inner.generate = timed("generate", inner.generate)
    inner.speech_tokenizer.decode = timed("decode", inner.speech_tokenizer.decode)
    codes_seen: list = []
    gen = inner.generate

    def keep_codes(*a, **k):
        out = gen(*a, **k)
        codes_seen.append(out[0][0].detach().cpu().numpy())
        return out

    inner.generate = keep_codes
    kw = {"do_sample": False, "subtalker_dosample": False} if args.greedy else {}

    def synth(text: str):
        return tts.generate_custom_voice(text=text, speaker=SPEAKER, language=LANGUAGE, **kw)

    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
    if args.limit:
        sentences = sentences[: args.limit]
    torch.manual_seed(0)
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        synth(text)
    out = REPO / "runs" / "bench"
    (out / args.name).parent.mkdir(parents=True, exist_ok=True)
    save_dir = REPO / "runs" / args.save_dir if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    info = {"model": MODELS[args.model], "dtype": args.dtype, "attn": args.attn, "greedy": args.greedy,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for seed in (int(x) for x in args.seeds.split(",")):
            for i, s in enumerate(sentences):
                torch.manual_seed(seed)
                stages.clear()
                codes_seen.clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                wavs, sr = synth(s["text"])
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                wav = np.asarray(wavs[0], dtype=np.float32)
                frames = int(codes_seen[0].shape[0])
                if save_dir:
                    stem = f"{Path(args.name).name}_s{seed}_{i:02d}"
                    sf.write(save_dir / f"{stem}.wav", wav, sr, subtype="PCM_16")
                    np.save(save_dir / f"{stem}.npy", codes_seen[0])
                row = {"engine": Path(args.name).name, "i": i, "seed": seed, "chars": len(s["text"]),
                       "total_sec": total, "stages": dict(stages), "audio_sec": len(wav) / sr, "frames": frames,
                       "sample_rate": sr, "finite": bool(np.isfinite(wav).all())}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    free, totalmem = torch.cuda.mem_get_info()
    info["device_used_gb"] = (totalmem - free) / 1e9
    (out / f"{args.name}.info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ALL DONE", info)


def bench_fast(tts: Qwen3TTSModel, args) -> None:
    from fast import Qwen3Fast, Sampling

    gen = {"do_sample": False, "subtalker_dosample": False} if args.greedy else {}
    probe = Qwen3Fast(tts, max_cache=args.max_cache)
    sampling = Sampling.from_kwargs(probe.prompt("準備ができました。", SPEAKER, LANGUAGE, **gen))
    del probe
    if args.engine == "kernels":
        from kernels import Qwen3Kernels

        eng = Qwen3Kernels(tts, sampling, max_cache=args.max_cache, wd=args.wd, sampler=args.sampler,
                           l2_keep_mb=args.l2_keep_mb, cp_pair={"auto": None, "on": True, "off": False}[args.cp_pair])
    else:
        eng = Qwen3Fast(tts, sampling, max_cache=args.max_cache, compile=args.compile)
    capture_sec = eng.capture()
    warm = {"prefill_sec": eng.warm_prefill(range(8, 65)), "decode_sec": eng.warm_decode()}
    torch.manual_seed(0)
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        eng.synth(text, SPEAKER, LANGUAGE, **gen)
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
    if args.limit:
        sentences = sentences[: args.limit]
    out = REPO / "runs" / "bench"
    (out / args.name).parent.mkdir(parents=True, exist_ok=True)
    save_dir = REPO / "runs" / args.save_dir if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    info = {"engine": args.engine, "wd": args.wd if args.engine == "kernels" else None,
            **({"sampler": eng.sampler, "l2_keep_mb": eng.l2_keep_mb, "l2_kept_bytes": eng.l2_kept_bytes, "cp_pair": eng.cp_pair}
               if args.engine == "kernels" else {}), "model": MODELS[args.model], "dtype": args.dtype, "attn": args.attn, "greedy": args.greedy,
            "max_cache": args.max_cache, "compile": args.compile, "capture_sec": capture_sec, "warm": warm, "sampling": vars(sampling),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for seed in (int(x) for x in args.seeds.split(",")):
            for i, s in enumerate(sentences):
                torch.manual_seed(seed)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                wav, sr, codes = eng.synth(s["text"], SPEAKER, LANGUAGE, **gen)
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                wav = np.asarray(wav, dtype=np.float32)
                if save_dir:
                    stem = f"{Path(args.name).name}_s{seed}_{i:02d}"
                    sf.write(save_dir / f"{stem}.wav", wav, sr, subtype="PCM_16")
                    np.save(save_dir / f"{stem}.npy", codes.cpu().numpy())
                st = dict(eng.stats)
                row = {"engine": Path(args.name).name, "i": i, "seed": seed, "chars": len(s["text"]),
                       "total_sec": total, "stages": {k: st[k] for k in ("prompt", "generate", "decode")},
                       "audio_sec": len(wav) / sr, "frames": int(codes.shape[0]), "sample_rate": sr,
                       "finite": bool(np.isfinite(wav).all()), "stopped": st["stopped"],
                       "captured": {k: st[k] for k in ("prefill_capture_sec", "decode_capture_sec") if k in st},
                       "cache_limited": st["cache_limited"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    free, totalmem = torch.cuda.mem_get_info()
    info["device_used_gb"] = (totalmem - free) / 1e9
    (out / f"{args.name}.info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ALL DONE", info)


if __name__ == "__main__":
    main()
