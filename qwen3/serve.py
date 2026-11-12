"""Qwen3-TTS（12Hz、CustomVoice）を、自作のカーネルのエンジン（qwen3/kernels.py）で動かす音声合成サーバー。

API は Irodori-TTS のサーバー（server/server.py）と同じ形にした（tools/bench_http.py でそのまま測れる）。
  GET  /health           準備ができたら 200。モデル、重みの型、話者の一覧を返す
  GET  /v1/audio/voices  話者の一覧（CustomVoice の決まった話者。参照音声からの声の複製はしない）
  GET  /v1/models
  GET  /v1/engine        起動のときの記録と、動いた回数（途中で CUDA Graph を記録した要求、上限で止めた要求の数）
  POST /v1/audio/speech  {"input": 文, "voice": 話者, "seed": 整数, "language": 言語, "response_format": "wav"}
       応答は 24kHz・16 ビットの WAV。ヘッダー X-Synthesis-Seconds（合成だけの時間）、X-Audio-Seconds（音声の長さ）
       seed を送ると、qwen3/bench.py --engine kernels で同じシードを使ったときと同じ音声になる。送らなければ毎回変わる
       Irodori のサーバーの speed と seconds（話す長さ）は受け付けない（このモデルは長さを指定できない）

環境変数
  QWEN3_MODEL               0.6b（既定）か 1.7b
  QWEN3_WD                  重みの型。i8g32（既定。int8 と 32 個ごとの尺度、speech.cpp の Q8_0 と同じ分け方）か bf16
  QWEN3_SPEAKER             既定の話者（既定 Ono_Anna）。QWEN3_LANGUAGE: 既定の言語（既定 Japanese）
  QWEN3_MAX_PROMPT_TOKENS   受け付けるプロンプトの長さの上限（既定 160。ふつうの日本語の 200 文字で約 124）。
                            起動のときに、6 からこの長さまでの最初の入力の CUDA Graph を記録する（1 つ約 0.1 秒）
  MAX_CHARS                 文字数の上限（既定 200）
  HOST、PORT                既定 0.0.0.0 と 18096（本番の :8088 とは別）

話し終えずに続く失敗: server/server.py と同じく、音声が 2.0 + 0.4 × 文字数 秒より長ければ 500 を返す（使う側が次のエンジンに回す）。
その長さに当たるフレーム数で、生成も止める（終わりのトークンを待たない）。

  /opt/qwen3/.venv/bin/python qwen3/serve.py     # イメージ sokutou/qwen3-dev:0.1.1。リポジトリを /work につなぐ
  イメージ sokutou/qwen3-server（scripts/build_qwen3.sh で作る）は、これを CMD で動かす。HF のキャッシュを HF_HOME につなぐ
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import MODELS, load  # noqa: E402
from fast import Qwen3Fast, Sampling  # noqa: E402
from kernels import Qwen3Kernels  # noqa: E402

log = logging.getLogger("qwen3-server")

MODEL = os.environ.get("QWEN3_MODEL", "0.6b")
WD = os.environ.get("QWEN3_WD", "i8g32")
SPEAKER = os.environ.get("QWEN3_SPEAKER", "Ono_Anna")
LANGUAGE = os.environ.get("QWEN3_LANGUAGE", "Japanese")
MAX_PROMPT = int(os.environ.get("QWEN3_MAX_PROMPT_TOKENS", "160"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "200"))
FRAME_RATE = 12.5  # 1 秒あたりのフレーム数（12Hz のコーデック。1 フレーム 1920 サンプル、24kHz）


def max_seconds(chars: int) -> float:
    """これより長い音声は、話し終えずに続いた失敗とみなす（server/server.py と同じ式）。"""
    return 2.0 + 0.4 * chars


class SpeechIn(BaseModel):
    input: str
    model: str = "qwen3-tts"
    voice: str = ""
    language: str = ""
    response_format: str = "wav"
    seed: int | None = None


class Server:
    def __init__(self) -> None:
        self.eng: Qwen3Kernels | None = None
        self.speakers: list[str] = []
        self.ready = False
        self.lock = asyncio.Lock()
        self.startup: dict = {}
        self.counts = {"requests": 0, "captured_during_request": 0, "frame_limited": 0, "rejected_long_prompt": 0}

    def load(self) -> None:
        if MODEL not in MODELS:
            raise RuntimeError(f"QWEN3_MODEL は {sorted(MODELS)} のどれか: {MODEL!r}")
        t0 = time.perf_counter()
        tts = load(MODEL, "bfloat16", "sdpa")
        self.speakers = sorted(tts.get_supported_speakers() or [])  # 公式の一覧は小文字（名前は大文字と小文字を区別しない）
        if SPEAKER.lower() not in self.speakers:
            raise RuntimeError(f"QWEN3_SPEAKER {SPEAKER!r} がない: {self.speakers}")
        probe = Qwen3Fast(tts, graphs=False)
        sampling = Sampling.from_kwargs(probe.prompt("準備ができました。", SPEAKER, LANGUAGE))
        del probe
        t1 = time.perf_counter()
        eng = Qwen3Kernels(tts, sampling, wd=WD)
        cap = eng.capture()
        warm_p = eng.warm_prefill(range(6, MAX_PROMPT + 1))
        warm_d = eng.warm_decode()
        # 試し合成（グラフを途中で記録しないことも確かめる）
        torch.manual_seed(0)
        for text in ("準備ができました。", "今日もよろしくお願いします。"):
            eng.synth(text, SPEAKER, LANGUAGE)
            if any(k in eng.stats for k in ("prefill_capture_sec", "decode_capture_sec")):
                raise RuntimeError(f"試し合成で CUDA Graph を記録した: {eng.stats}")
        free, total = torch.cuda.mem_get_info()
        self.startup = {"model": MODELS[MODEL], "wd": WD, "sampler": eng.sampler, "l2_kept_mib": eng.l2_kept_bytes / 2**20,
                        "load_sec": t1 - t0, "capture_sec": cap, "warm_prefill_sec": warm_p, "warm_decode_sec": warm_d,
                        "total_sec": time.perf_counter() - t0, "max_prompt_tokens": MAX_PROMPT,
                        "device": torch.cuda.get_device_name(0), "device_used_gb": (total - free) / 1e9}
        self.eng = eng
        self.ready = True
        log.info("ready: %s", self.startup)

    def synthesize(self, text: str, speaker: str, language: str, seed: int | None) -> tuple[np.ndarray, int, dict]:
        """Qwen3Fast.synth と同じ順（プロンプト、生成、復号）。プロンプトの長さが上限を超えたら TooLong を投げる。"""
        eng = self.eng
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
        eng.stats = {}
        kw = eng.prompt(text, speaker, language)
        n = int(kw["inputs_embeds"].shape[1])
        if n > MAX_PROMPT:
            raise TooLong(n)
        cap = math.ceil(max_seconds(len(text)) * FRAME_RATE) + 1
        codes = eng.generate_codes(kw, max_frames=cap)
        wav, sr = eng.decode(codes)
        torch.cuda.synchronize()
        st = dict(eng.stats)
        st["frames"] = int(codes.shape[0])
        st["prompt_tokens"] = n
        return np.asarray(wav, dtype=np.float32), sr, st


class TooLong(Exception):
    def __init__(self, n: int) -> None:
        super().__init__(f"{n} prompt tokens")
        self.n = n


srv = Server()


async def lifespan(app):
    await asyncio.to_thread(srv.load)
    yield


app = FastAPI(title="qwen3-tts-server", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    body = {"ok": srv.ready, "model": MODELS.get(MODEL, [MODEL])[0], "wd": WD, "voices": srv.speakers,
            "default_voice": SPEAKER, "language": LANGUAGE}
    return JSONResponse(body, status_code=200 if srv.ready else 503)


@app.get("/v1/audio/voices")
async def voices() -> dict:
    return {"voices": [{"name": s} for s in srv.speakers]}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "qwen3-tts", "object": "model"}]}


@app.get("/v1/engine")
async def engine_info() -> dict:
    return {"startup": srv.startup, **srv.counts}


def to_wav(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")  # qwen3/bench.py の保存と同じ変換
    return buf.getvalue()


@app.post("/v1/audio/speech")
async def speech(body: SpeechIn) -> Response:
    if not srv.ready:
        raise HTTPException(503, "model is loading")
    if body.response_format != "wav":
        raise HTTPException(400, "only response_format=wav is supported")
    text = body.input.strip()
    if not text:
        raise HTTPException(400, "input is empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"input is longer than {MAX_CHARS} characters")
    speaker = body.voice or SPEAKER
    if speaker.lower() not in srv.speakers:
        raise HTTPException(404, f"unknown voice {speaker!r}; available: {srv.speakers}")
    language = body.language or LANGUAGE
    started = time.perf_counter()  # server/server.py と同じく、前の合成を待つ時間も X-Synthesis-Seconds に入れる
    async with srv.lock:
        try:
            audio, sr, st = await asyncio.to_thread(srv.synthesize, text, speaker, language, body.seed)
        except TooLong as e:
            srv.counts["rejected_long_prompt"] += 1
            raise HTTPException(400, f"input is too long ({e.n} prompt tokens > {MAX_PROMPT})") from e
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise HTTPException(503, f"out of GPU memory: {e}") from e
        except ValueError as e:  # 言語の名前が違うなど（公式の generate_custom_voice の確かめ）
            raise HTTPException(400, str(e)) from e
    elapsed = time.perf_counter() - started
    srv.counts["requests"] += 1
    captured = [k for k in ("prefill_capture_sec", "decode_capture_sec") if k in st]
    if captured:
        srv.counts["captured_during_request"] += 1
        log.error("CUDA Graph を要求の途中で記録した: %s（%d 文字）", captured, len(text))
    duration = len(audio) / sr
    if st.get("frame_limited") or duration > max_seconds(len(text)):
        srv.counts["frame_limited"] += 1
        raise HTTPException(500, f"implausible duration {duration:.1f}s for {len(text)} chars (frames {st['frames']})")
    log.info("voice=%s chars=%d tokens=%d frames=%d audio=%.2fs synth=%.3fs", speaker, len(text), st["prompt_tokens"], st["frames"],
             duration, elapsed)
    return Response(to_wav(audio, sr), media_type="audio/wav",
                    headers={"X-Synthesis-Seconds": f"{elapsed:.3f}", "X-Audio-Seconds": f"{duration:.3f}"})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "18096")), log_level="info")
