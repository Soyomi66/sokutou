"""音声合成サーバー（POST /v1/audio/speech）に 48 文 × シード 1, 2 を送って測る。

本文（voice、input、seconds、seed。--no-seconds なら seconds を除く）を送り、サーバーが返す X-Synthesis-Seconds（合成だけの時間）と、
HTTP を含めた時間を記録する。X-Synthesis-Seconds を返さないサーバー（speech.cpp など）では、total_sec に HTTP を含めた時間を入れる。
標準ライブラリだけで動く。

  python3 tools/bench_http.py --url http://localhost:8090 --name engine-http

  python3 tools/bench_http.py --url http://127.0.0.1:18092 --name speechcpp --voice <声の名前> --extra '{"sway_coeff": -1}'

--voice を省くと、サーバーの既定の声で話す（server.py は名前順で最初の声）。
サーバーの既定のポート（:8088）には、--allow-production を付けたときだけ送る（使っているサーバーを誤って測らないため）。
出力: runs/bench/<name>.jsonl（tools/bench_summary.py で集計できる形）
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def post(url: str, body: dict):
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers  # 名前の大文字と小文字を区別しない


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--voice", default="", help="声の名前（既定はサーバーの既定の声）")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--extra", default="{}", help="本文に足す JSON（例: speech.cpp には {\"sway_coeff\": -1}）")
    ap.add_argument("--no-seed", action="store_true",
                    help="seed を送らない（サーバーが乱数で決める）。--seeds は回数だけに使う")
    ap.add_argument("--no-seconds", action="store_true",
                    help="seconds を送らない（話す長さを受け付けないモデル。speech.cpp の Qwen3-TTS など）")
    ap.add_argument("--allow-production", action="store_true",
                    help=":8088 のサーバーにも送る（使っているサーバーを測るときだけ付ける）")
    ap.add_argument("--keep-going", action="store_true",
                    help="エラーの応答（話し終えずに続いた音声を断る 500 など）を status として記録し、次の文に進む")
    ap.add_argument("--save-dir", default="", help="返ってきた WAV を runs/ の下のこのディレクトリに <name>_s<シード>_<番号>.wav で保存する")
    args = ap.parse_args()
    extra = json.loads(args.extra)
    if args.url.rstrip("/").endswith(":8088") and not args.allow_production:
        raise SystemExit(":8088 のサーバーには、--allow-production を付けたときだけ送る")
    sentences = json.loads((REPO / "data" / "sentences.json").read_text(encoding="utf-8"))
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        post(args.url, {"input": text, "voice": args.voice, **({} if args.no_seconds else {"seconds": 2.0}), **extra})
    out = REPO / "runs" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    save_dir = REPO / "runs" / args.save_dir if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for seed in (int(x) for x in args.seeds.split(",")):
            for i, s in enumerate(sentences):
                t0 = time.perf_counter()
                body = {"input": s["text"], "voice": args.voice, **({} if args.no_seconds else {"seconds": s["seconds"]}), **extra}
                if not args.no_seed:
                    body["seed"] = seed
                try:
                    wav, headers = post(args.url, body)
                except urllib.error.HTTPError as e:
                    if not args.keep_going:
                        raise
                    detail = e.read().decode("utf-8", "replace")[:300]
                    f.write(json.dumps({"engine": args.name, "i": i, "seed": None if args.no_seed else seed,
                                        "chars": len(s["text"]), "status": e.code, "detail": detail,
                                        "wall_sec": time.perf_counter() - t0}, ensure_ascii=False) + "\n")
                    print(f"seed {seed} 文 {i}: HTTP {e.code} {detail}", flush=True)
                    continue
                wall = time.perf_counter() - t0
                if save_dir:
                    (save_dir / f"{Path(args.name).name}_s{seed}_{i:02d}.wav").write_bytes(wav)
                with wave.open(io.BytesIO(wav)) as w:
                    sr = w.getframerate()
                    audio_sec = w.getnframes() / sr
                row = {"engine": args.name, "i": i, "seed": None if args.no_seed else seed, "chars": len(s["text"]), "status": 200,
                       "seconds": s["seconds"],
                       "total_sec": float(headers.get("X-Synthesis-Seconds", wall)),
                       "has_synthesis_header": "X-Synthesis-Seconds" in headers,
                       "wall_sec": wall, "audio_sec": audio_sec, "sample_rate": sr}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("ALL DONE")


if __name__ == "__main__":
    main()
