"""qwen3/fast.py のエンジンを公式の実装と比べる。イメージ sokutou/qwen3-official の中で動かす。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-official:0.1.1 scripts/run_in_image.sh \
    qwen3/check.py --model 0.6b --name qwen3/check-0.6b --limit 8

確かめること
  1. 公式の generate が使う logits processor の並び（言語モデルと code predictor）。エンジンの書き方の前提を確かめる
  2. エンジンの multinomial（argmax(p / q)、q ~ Exp(1)）が torch.multinomial と同じ番号を引き、乱数を同じだけ進めるか
  3. 同じ文・同じシードで、公式とエンジンのコードがどこまで一致するか（いちばん確率の高いトークンを選ぶ設定と、
     サンプリングありの設定）。bfloat16 では注意の計算の順が違うので、どこかで分かれることがある。float32 で
     ほぼ最後まで一致すれば、処理の順と乱数の使い方は公式と同じと判断する
出力: runs/check/<name>.jsonl と、標準出力の要約
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import LANGUAGE, SPEAKER, load  # noqa: E402
from fast import Qwen3Fast, Sampling, _multinomial  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GREEDY = {"do_sample": False, "subtalker_dosample": False}


def inspect_processors(tts) -> dict:
    """公式の generate を 1 回動かし、言語モデルと code predictor の logits processor の並びを記録する。"""
    seen: dict = {}
    for name, mod in (("talker", tts.model.talker), ("code_predictor", tts.model.talker.code_predictor)):
        orig = mod._get_logits_processor

        def wrap(*a, _orig=orig, _name=name, **k):
            procs = _orig(*a, **k)
            if _name not in seen:
                seen[_name] = [{"class": type(p).__name__, **{kk: (vv if isinstance(vv, (int, float, bool, str)) else
                                                                 str(vv)[:80]) for kk, vv in vars(p).items()
                                                               if not kk.startswith("_")}} for p in procs]
            return procs

        mod._get_logits_processor = wrap
    torch.manual_seed(0)
    tts.generate_custom_voice(text="確かめます。", speaker=SPEAKER, language=LANGUAGE)
    for mod in (tts.model.talker, tts.model.talker.code_predictor):
        del mod._get_logits_processor
    seen["code_predictor_generation_config"] = tts.model.talker.code_predictor.generation_config.to_diff_dict()
    return seen


def check_multinomial(device: str, n: int = 2000) -> dict:
    """同じシードで torch.multinomial と _multinomial を引き比べる。そのあとの乱数も同じかを見る（進み方が同じか）。"""
    same = same_after = 0
    for seed in range(n):
        g = torch.Generator(device=device).manual_seed(10_000 + seed)
        logits = torch.randn((1, 3072 if seed % 2 else 2048), generator=g, device=device) * 3
        probs = torch.softmax(logits, dim=-1)
        torch.manual_seed(seed)
        a = torch.multinomial(probs, num_samples=1)
        ra = torch.rand(4, device=device)
        torch.manual_seed(seed)
        b = _multinomial(probs)
        rb = torch.rand(4, device=device)
        same += int(torch.equal(a, b))
        same_after += int(torch.equal(ra, rb))
    return {"draws": n, "same_index": same, "same_rng_after": same_after}


def official_codes(tts, text: str, seed: int, gen: dict) -> torch.Tensor:
    inner = tts.model
    kept: list = []
    orig = inner.generate

    def keep(*a, **k):
        out = orig(*a, **k)
        kept.append(out[0][0])
        return out

    inner.generate = keep
    try:
        torch.manual_seed(seed)
        tts.generate_custom_voice(text=text, speaker=SPEAKER, language=LANGUAGE, **gen)
    finally:
        del inner.generate
    return kept[0]


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    n = min(a.shape[0], b.shape[0])
    eq = (a[:n] == b[:n]).all(dim=1)
    first = int((~eq).nonzero()[0, 0]) if bool((~eq).any()) else n
    return {"frames_official": int(a.shape[0]), "frames_engine": int(b.shape[0]), "first_diff_frame": first,
            "identical": bool(a.shape[0] == b.shape[0] and bool(eq.all())),
            "frac_equal_frames": float(eq.float().mean()) if n else 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--max-cache", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0", help="cpu で、CUDA Graph を使わない経路を確かめられる")
    ap.add_argument("--no-graph", action="store_true", help="CUDA Graph に記録せず、1 つずつ動かす")
    ap.add_argument("--no-inspect", action="store_true")
    ap.add_argument("--compile", action="store_true", help="1 フレーム分を torch.compile してから記録する")
    ap.add_argument("--engine", choices=["fast", "kernels"], default="fast",
                    help="fast: HF のモジュールと CUDA Graph（qwen3/fast.py）、kernels: 自作のカーネル（qwen3/kernels.py）")
    ap.add_argument("--wd", default="", help="--engine kernels の重みの型（bf16、i8g32、f32。既定はモデルの型）")
    ap.add_argument("--sampler", choices=["torch", "fused"], default="torch",
                    help="--engine kernels のサンプリング。公式と突き合わせるのは torch（fused は乱数が別なので一致しない）")
    ap.add_argument("--max-new-tokens", type=int, default=400,
                    help="公式とエンジンの両方に渡す上限（公式の既定は 8192。終わりのトークンが出ないと長く続くため）")
    args = ap.parse_args()
    if args.device == "cpu":
        args.no_graph = True

    tts = load(args.model, args.dtype, "sdpa", args.device)
    dev = torch.device(args.device)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    out = REPO / "runs" / "check" / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = {} if args.no_inspect else {"processors": inspect_processors(tts),
                                       "multinomial": check_multinomial(dev.type)}
    print(json.dumps(head, ensure_ascii=False, indent=1, default=str))
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))[: args.limit]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"head": head, "args": vars(args)}, ensure_ascii=False, default=str) + "\n")
        for mode in args.modes.split(","):
            gen = {**(GREEDY if mode == "greedy" else {}), "max_new_tokens": args.max_new_tokens}
            probe = Qwen3Fast(tts, max_cache=args.max_cache)
            kw = probe.prompt(sentences[0]["text"], SPEAKER, LANGUAGE, **gen)
            if args.engine == "kernels":
                from kernels import Qwen3Kernels

                wd = args.wd or ("f32" if args.dtype == "float32" else "bf16")
                eng = Qwen3Kernels(tts, Sampling.from_kwargs(kw), max_cache=args.max_cache, wd=wd, sampler=args.sampler)
            else:
                eng = Qwen3Fast(tts, Sampling.from_kwargs(kw), max_cache=args.max_cache, compile=args.compile)
            del probe
            cap = 0.0 if args.no_graph else eng.capture()
            print(f"[{mode}] capture {cap:.2f}s graph={not args.no_graph} sampling={eng.s}", flush=True)
            for seed in (seeds if mode == "sample" else seeds[:1]):
                for i, s in enumerate(sentences):
                    ref = official_codes(tts, s["text"], seed, gen)
                    torch.manual_seed(seed)
                    kw = eng.prompt(s["text"], SPEAKER, LANGUAGE, **gen)
                    sync()
                    t0 = time.perf_counter()
                    got = eng.generate_codes(kw)
                    sync()
                    dt = time.perf_counter() - t0
                    row = {"mode": mode, "seed": seed, "i": i, "dtype": args.dtype, **compare(ref, got),
                           "engine_generate_sec": dt, "engine_ms_per_frame": 1000 * dt / max(1, got.shape[0]),
                           **{k: eng.stats[k] for k in ("prompt_tokens", "stopped", "cache_limited")}}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(row, flush=True)
            del eng
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    for mode in args.modes.split(","):
        r = [x for x in rows if x["mode"] == mode]
        if r:
            print(f"[{mode}] identical {sum(x['identical'] for x in r)}/{len(r)}, "
                  f"first_diff median {sorted(x['first_diff_frame'] for x in r)[len(r) // 2]}, "
                  f"ms/frame median {sorted(x['engine_ms_per_frame'] for x in r)[len(r) // 2]:.2f}")


if __name__ == "__main__":
    main()
