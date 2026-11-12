"""Fish Audio Drama 3 の公式の実装（fish-speech）を、48 文 × シードで測る。イメージ sokutou/drama3-official の中で動かす。

  PYTHON=/opt/fish-speech/.venv/bin/python IMAGE=sokutou/drama3-official:0f8a12e scripts/run_in_image.sh \
    drama3/bench.py --name drama3/official-1

fish-speech の API サーバーと同じ順に呼ぶ: init_model（bf16、エキスパートは FP8）と load_codec_model（bf16）で読み込み、
1 文ごとに set_seed、generate_long_drama3（文を区切り、1 区切りずつ生成）、コーデックの from_indices（bf16 の autocast）。
1 文の時間は、文を渡してから 44.1kHz の波形（CPU の numpy）ができるまで。内訳は prefill（最初の入力）、generate
（prefill を含む生成）、decode（復号）。
--engine kernels では、generate_long_drama3 が呼ぶ generate_drama3 を drama3/kernels.py の Drama3Engine.generate に
差し替える（文の区切り方と、前の区切りをプロンプトに入れるやり方は公式のまま）。--wd i8g32 で、エキスパートのほかの重みを
int8 にする。--prefill-graphs と --prefix-cache は最初の入力を速くする（drama3/kernels.py の Drama3Engine.prefill）。
--drop-codec-masks で、コーデックの読まれない因果マスク（3 GiB）を空ける。
サンプリングは --temperature 1.0、--top-p 0.9、--top-k 30（モデルカードと speech.cpp の既定。fish-speech の既定は
0.7・0.7・30 なので、明示して渡す）。main token の繰り返しを引き直す処理は fish-speech の既定（RAS、reference_sampling なし）。
--ref と --ref-text で参照を 1 本使う（コーデックで符号化したコードを使い回す。符号化の時間は 1 文の時間に入れない。
fish-speech のサーバーも参照のコードをメモリーに残す）。参照なしでは、モデルが選ぶ声で話す。
出力: runs/bench/<name>.jsonl と <name>.info.json（GPU のメモリー）、--save-dir で WAV とコード（[10, フレーム数] の .npy）
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
REPO_ID, REVISION = "fishaudio/drama-3", "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
SAMPLE_RATE = 44100


def checkpoint_path() -> Path:
    from huggingface_hub import constants, snapshot_download

    # HF_HUB_OFFLINE=1（イメージの既定）ならキャッシュだけを使い、0 なら取得する（この版はオフラインでキャッシュに戻らない）
    return Path(snapshot_download(REPO_ID, revision=REVISION, local_files_only=constants.HF_HUB_OFFLINE))


def gib_used() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="1 区切りのフレームの上限（サーバーの既定 1024）")
    ap.add_argument("--ref", default="", help="参照の WAV（runs/ からの相対、または絶対のパス）")
    ap.add_argument("--ref-text", default="", help="参照の書き起こし")
    ap.add_argument("--save-dir", default="", help="runs/ の下のこのディレクトリに WAV（44.1kHz）とコードを保存する")
    ap.add_argument("--limit", type=int, default=0, help="最初の N 文だけ（試すとき）")
    ap.add_argument("--engine", choices=("official", "kernels"), default="official",
                    help="kernels: 1 フレームを自作のカーネル（drama3/kernels.py）で作る。イメージは sokutou/drama3-dev")
    ap.add_argument("--no-graph", action="store_true", help="kernels で CUDA Graph を使わない（確かめるとき）")
    ap.add_argument("--wd", choices=("bf16", "i8g32"), default="bf16",
                    help="kernels のエキスパートのほかの重みの型（i8g32: int8 と 32 個ごとの尺度）")
    ap.add_argument("--prefill-graphs", type=int, default=0,
                    help="kernels: 最初の入力を区切りごとに CUDA Graph にして、新しい順にこの数まで持つ（0: eager）")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="kernels: 前の要求と頭が同じ行（参照）の Gated DeltaNet の状態を写して使い回す")
    ap.add_argument("--drop-codec-masks", action="store_true",
                    help="コーデックの読まれない因果マスク（3 GiB）を空ける（drama3/kernels.py の drop_unused_codec_masks）")
    return ap


def main() -> None:
    args = parser().parse_args()
    from fish_speech.models.text2semantic import inference as fi

    info: dict = {}
    path = checkpoint_path()
    info["gib_before_load"] = gib_used()
    t0 = time.perf_counter()
    model, _ = fi.init_model(str(path), "cuda:0", torch.bfloat16)
    info["load_model_sec"] = time.perf_counter() - t0
    info["gib_after_model"] = gib_used()
    t0 = time.perf_counter()
    codec = fi.load_codec_model(str(path / "codec.pth"), "cuda:0", torch.bfloat16)
    info["load_codec_sec"] = time.perf_counter() - t0
    info["gib_after_codec"] = gib_used()
    run(args, model, codec, info)


def run(args, model, codec, info: dict, engine=None) -> None:
    """読み込んだモデルとコーデックで測る。engine を渡せば、作ってあるエンジン（Drama3Engine）を使う。"""
    from fish_speech.models.text2semantic import inference as fi
    from fish_speech.utils import set_seed

    info.update({"repo": [REPO_ID, REVISION], "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                 "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                              "max_new_tokens": args.max_new_tokens, "reference_sampling": None},
                 "cuda_graph": __import__("os").environ.get("FISH_DRAMA3_CUDA_GRAPH", "1")})

    # generate_drama3 が loguru に出す prefill の時間を拾う
    prefill: list[float] = []

    def sink(message) -> None:
        m = re.search(r"Drama 3 prefill: (\d+) rows in ([0-9.]+)s", message.record["message"])
        if m:
            prefill.append(float(m.group(2)))
            rows.append(int(m.group(1)))

    rows: list[int] = []
    logger.remove()
    logger.add(sink, level="INFO")
    logger.add(lambda m: print(m, end=""), level="WARNING")

    stages: dict[str, float] = {}
    gen, eng = fi.generate_drama3, None
    info["engine"] = args.engine
    if args.engine == "kernels":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kernels import Drama3Engine

        eng = engine if engine is not None else Drama3Engine(model, wd=args.wd, graphs=not args.no_graph,
                                                              prefill_graphs=args.prefill_graphs,
                                                              prefix_cache=args.prefix_cache)
        gen = eng.generate
        info.update(wd=eng.wd, prefill_graphs=eng.pf_capacity, prefix_cache=eng.prefix_cache)
        info["engine_setup_sec"] = eng.setup_sec
        info["gib_after_engine"] = gib_used()

    if args.drop_codec_masks:
        from kernels import drop_unused_codec_masks

        info["codec_masks_freed_gib"] = drop_unused_codec_masks(codec) / 2**30

    def timed_generate(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = gen(*a, **k)
        torch.cuda.synchronize()
        stages["generate"] = stages.get("generate", 0.0) + time.perf_counter() - t
        return out

    fi.generate_drama3 = timed_generate

    prompt_text: list[str] = []
    prompt_tokens: list[torch.Tensor] = []
    if args.ref:
        ref = Path(args.ref) if Path(args.ref).is_absolute() else REPO / "runs" / args.ref
        audio, sr = sf.read(ref, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.mean(1))
        if sr != SAMPLE_RATE:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        with torch.inference_mode():
            x = audio.to("cuda:0", dtype=next(codec.parameters()).dtype)[None, None]
            codes = codec.encode(x, torch.tensor([x.shape[-1]], device="cuda:0"))[0][0]
        prompt_text, prompt_tokens = [args.ref_text], [codes.cpu()]
        info["reference"] = {"file": str(args.ref), "seconds": len(audio) / SAMPLE_RATE, "frames": int(codes.shape[-1])}

    @torch.inference_mode()
    def synth(text: str):
        frames, waves = 0, []
        for resp in fi.generate_long_drama3(model=model, text=text, max_new_tokens=args.max_new_tokens,
                                             top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
                                             prompt_text=prompt_text, prompt_tokens=prompt_tokens):
            if resp.action != "sample":
                continue
            frames += int(resp.codes.shape[1])
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wave = codec.from_indices(resp.codes.to("cuda:0")[None])[0].squeeze()
            waves.append(wave.float().cpu().numpy())
            stages["decode"] = stages.get("decode", 0.0) + time.perf_counter() - t
            codes_seen.append(resp.codes.numpy())
        return np.concatenate(waves) if waves else np.zeros(0, np.float32), frames

    codes_seen: list[np.ndarray] = []
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
    if args.limit:
        sentences = sentences[: args.limit]
    # 最初の要求で CUDA Graph を記録する（サーバーの warm_up と同じく、測る前にすませる）
    t0 = time.perf_counter()
    set_seed(0)
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        synth(text)
    info["warmup_sec"] = time.perf_counter() - t0
    info["gib_after_warmup"] = gib_used()
    out = REPO / "runs" / "bench"
    (out / args.name).parent.mkdir(parents=True, exist_ok=True)
    save_dir = REPO / "runs" / args.save_dir if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for seed in (int(x) for x in args.seeds.split(",")):
            for i, s in enumerate(sentences):
                stages.clear()
                prefill.clear()
                rows.clear()
                codes_seen.clear()
                set_seed(seed)
                captures = getattr(eng, "pf_captures", 0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                wav, frames = synth(s["text"])
                total = time.perf_counter() - t0
                if save_dir:
                    stem = f"{Path(args.name).name}_s{seed}_{i:02d}"
                    sf.write(save_dir / f"{stem}.wav", wav, SAMPLE_RATE, subtype="PCM_16")
                    if codes_seen:
                        np.save(save_dir / f"{stem}.npy", np.concatenate(codes_seen, axis=1))
                row = {"engine": Path(args.name).name, "i": i, "seed": seed, "chars": len(s["text"]),
                       "total_sec": total, "stages": {**stages, "prefill": sum(prefill)}, "audio_sec": len(wav) / SAMPLE_RATE,
                       "frames": frames, "chunks": len(codes_seen), "prompt_rows": rows, "sample_rate": SAMPLE_RATE,
                       "finite": bool(np.isfinite(wav).all()),
                       "prefill_captures": getattr(eng, "pf_captures", 0) - captures}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    info["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    info["gib_after_bench"] = gib_used()
    (out / f"{args.name}.info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ALL DONE", info)


if __name__ == "__main__":
    main()
