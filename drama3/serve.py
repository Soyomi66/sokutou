"""Fish Audio Drama 3 を、自作のカーネルのエンジン（drama3/kernels.py）で動かす音声合成サーバー。

API は Irodori-TTS と Qwen3-TTS のサーバー（server/server.py、qwen3/serve.py）と同じ形にした（tools/bench_http.py でそのまま測れる）。
  GET  /health           準備ができたら 200。モデル、重みの型、声の一覧を返す
  GET  /v1/audio/voices  声の一覧（none は参照なし。文ごとにモデルが声を選ぶ）
  GET  /v1/models
  GET  /v1/engine        起動のときの記録と、動いた回数
  POST /v1/audio/speech  {"input": 文, "voice": 声, "seed": 整数, "response_format": "wav"}
       応答は 44.1kHz・16 ビットの WAV。ヘッダー X-Synthesis-Seconds（合成の時間。前の合成を待つ時間を含む）、
       X-Audio-Seconds（音声の長さ）
       seed を送ると、drama3/bench.py --engine kernels で同じシード・同じ設定を使ったときと同じ音声になる。参照ありでは、
       前の要求と同じ参照で（違う参照のあとの 1 回目は、最初の入力の区切り方が変わるので、少し違う音声になる）、
       その声が名前順で最初の声のとき（下の「声」）。

声（参照）: DRAMA3_VOICES のディレクトリの <声の名前>/ に、WAV（*.wav）と、同じ名前の書き起こし（*.lab、UTF-8）を置く
（fish-speech の references/ と同じ形）。起動のときに名前順にコーデックで符号化し、コードをメモリーに残す。
コーデックの Snake（descript-audio-codec の snake）は TorchScript なので、プロセスで最初の符号化だけ、2 回目以降と値が
少し違う（人の声の参照で、1,700 個の符号のうち 289 個）。名前順で最初の声だけが、bench.py --ref（参照を最初に符号化する）と
同じコードになる。
参照の行の使い回し（prefix cache）は 1 つの声の分だけを持つ。声を切り替えた最初の要求は、参照の行もすべて流す。

環境変数
  DRAMA3_WD               エキスパートのほかの重みの型。i8g32（既定。int8 と 32 個ごとの尺度）か bf16
  DRAMA3_VOICE            既定の声（既定 none）
  DRAMA3_VOICES           声のディレクトリ（既定 /voices）
  DRAMA3_PREFILL_GRAPHS   最初の入力の CUDA Graph を新しい順にこの数まで持つ（既定 24。1 つ約 80 MiB。0 なら eager。
                          2 回目に出た区切りから記録する）
  DRAMA3_PREFIX_CACHE     1（既定）なら、同じ参照の行の状態を使い回す
  DRAMA3_TEMPERATURE、DRAMA3_TOP_P、DRAMA3_TOP_K   サンプリング（既定 1.0、0.9、30。モデルカードと speech.cpp の既定）
  MAX_CHARS               文字数の上限（既定 200）
  HOST、PORT              既定 0.0.0.0 と 18098（本番の :8088 とは別）

話し終えずに続く失敗: server/server.py と同じく、音声が 2.0 + 0.4 × 文字数 秒より長ければ 500 を返す（使う側が次のエンジンに回す）。
生成も、区切りをまたいだフレームの合計がその長さを超えたところで止める（終わりのトークンを待たない）。

  /opt/fish-speech/.venv/bin/python drama3/serve.py   # イメージ sokutou/drama3-server（docker/drama3.Dockerfile）。HF のキャッシュを HF_HOME に
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
from bench import REPO_ID, REVISION, SAMPLE_RATE, checkpoint_path  # noqa: E402
from kernels import Drama3Engine, drop_unused_codec_masks  # noqa: E402

log = logging.getLogger("drama3-server")

WD = os.environ.get("DRAMA3_WD", "i8g32")
VOICE = os.environ.get("DRAMA3_VOICE", "none")
VOICES = Path(os.environ.get("DRAMA3_VOICES", "/voices"))
PREFILL_GRAPHS = int(os.environ.get("DRAMA3_PREFILL_GRAPHS", "24"))
PREFIX_CACHE = os.environ.get("DRAMA3_PREFIX_CACHE", "1") == "1"
TEMPERATURE = float(os.environ.get("DRAMA3_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("DRAMA3_TOP_P", "0.9"))
TOP_K = int(os.environ.get("DRAMA3_TOP_K", "30"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "200"))


def max_seconds(chars: int) -> float:
    """これより長い音声は、話し終えずに続いた失敗とみなす（server/server.py と同じ式）。"""
    return 2.0 + 0.4 * chars


class SpeechIn(BaseModel):
    input: str
    model: str = "drama-3"
    voice: str = ""
    response_format: str = "wav"
    seed: int | None = None


class Server:
    def __init__(self) -> None:
        self.model = self.codec = self.eng = self.fi = None
        self.voices: dict[str, tuple[list[str], list[torch.Tensor]]] = {}
        self.ready = False
        self.lock = asyncio.Lock()
        self.startup: dict = {}
        self.counts = {"requests": 0, "frame_limited": 0, "prefill_captures": 0}

    def load(self) -> None:
        from fish_speech.models.text2semantic import inference as fi

        self.fi = fi
        t0 = time.perf_counter()
        path = checkpoint_path()
        self.model, _ = fi.init_model(str(path), "cuda:0", torch.bfloat16)
        self.codec = fi.load_codec_model(str(path / "codec.pth"), "cuda:0", torch.bfloat16)
        freed = drop_unused_codec_masks(self.codec)
        self.hop = int(self.codec.frame_length)  # 1 フレームのサンプル数（2048。hop_length 512 × 量子化の前の間引き 4）
        assert self.hop == 2048, self.hop
        assert int(self.codec.sample_rate) == SAMPLE_RATE
        t1 = time.perf_counter()
        self.voices = {"none": ([], [])}
        if VOICES.is_dir():
            for d in sorted(p for p in VOICES.iterdir() if p.is_dir()):
                refs = [(w, w.with_suffix(".lab")) for w in sorted(d.glob("*.wav")) if w.with_suffix(".lab").is_file()]
                if refs:
                    self.voices[d.name] = ([lab.read_text(encoding="utf-8").strip() for _, lab in refs],
                                           [self.encode(w) for w, _ in refs])
        if VOICE not in self.voices:
            raise RuntimeError(f"DRAMA3_VOICE {VOICE!r} がない: {sorted(self.voices)}")
        t2 = time.perf_counter()
        eng = Drama3Engine(self.model, wd=WD, graphs=True, prefill_graphs=PREFILL_GRAPHS, prefix_cache=PREFIX_CACHE)
        cap = eng.capture()
        self.eng = eng
        fi.generate_drama3 = eng.generate  # generate_long_drama3（文の区切り方とプロンプトの作り方は公式のまま）が呼ぶ
        # 試し合成: 既定の声で 2 回（2 回目で参照の行の状態を写す）
        for text in ("準備ができました。", "今日もよろしくお願いします。"):
            self.synthesize(text, VOICE, 0)
        free, total = torch.cuda.mem_get_info()
        self.startup = {"model": [REPO_ID, REVISION], "wd": WD, "voices": sorted(self.voices), "default_voice": VOICE,
                        "prefill_graphs": PREFILL_GRAPHS, "prefix_cache": PREFIX_CACHE,
                        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K},
                        "codec_masks_freed_gib": freed / 2**30, "load_sec": t1 - t0, "encode_voices_sec": t2 - t1,
                        "capture_sec": cap, "total_sec": time.perf_counter() - t0,
                        "device": torch.cuda.get_device_name(0), "device_used_gb": (total - free) / 1e9}
        self.ready = True
        log.info("ready: %s", self.startup)

    @torch.inference_mode()
    def encode(self, wav: Path) -> torch.Tensor:
        """参照の WAV をコードにする（drama3/bench.py の --ref と同じ。fish-speech の encode_reference と同じ手順）。"""
        audio, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = torch.from_numpy(audio.mean(1))
        if sr != SAMPLE_RATE:
            import torchaudio

            x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
        x = x.to("cuda:0", dtype=next(self.codec.parameters()).dtype)[None, None]
        return self.codec.encode(x, torch.tensor([x.shape[-1]], device="cuda:0"))[0][0].cpu()

    @torch.inference_mode()
    def synthesize(self, text: str, voice: str, seed: int | None) -> tuple[np.ndarray, dict]:
        """drama3/bench.py の synth と同じ順（generate_long_drama3、区切りごとに from_indices）。"""
        from fish_speech.utils import set_seed

        if seed is None:
            torch.seed()
        else:
            set_seed(seed)
        prompt_text, prompt_tokens = self.voices[voice]
        limit = math.ceil(max_seconds(len(text)) * SAMPLE_RATE / self.hop) + 1  # この数のフレームで止める
        captures = self.eng.pf_captures
        frames, waves, chunks = 0, [], 0
        for resp in self.fi.generate_long_drama3(model=self.model, text=text, max_new_tokens=limit, top_p=TOP_P, top_k=TOP_K,
                                                 temperature=TEMPERATURE, prompt_text=prompt_text,
                                                 prompt_tokens=prompt_tokens):
            if resp.action != "sample":
                continue
            frames += int(resp.codes.shape[1])
            chunks += 1
            if frames > limit:  # max_new_tokens は区切りごとの上限なので、区切りをまたいだ合計もここで止める（呼ぶ側が 500 を返す）
                break
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wave = self.codec.from_indices(resp.codes.to("cuda:0")[None])[0].squeeze()
            waves.append(wave.float().cpu().numpy())
        torch.cuda.synchronize()
        audio = np.concatenate(waves) if waves else np.zeros(0, np.float32)
        # limited: 合計の上限で止めた（最後の区切りは復号していないので、audio の長さでは判定できない）
        return audio, {"frames": frames, "chunks": chunks, "limited": frames > limit,
                       "prefill_captures": self.eng.pf_captures - captures}


srv = Server()


async def lifespan(app):
    await asyncio.to_thread(srv.load)
    yield


app = FastAPI(title="drama3-tts-server", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    body = {"ok": srv.ready, "model": REPO_ID, "wd": WD, "voices": sorted(srv.voices), "default_voice": VOICE}
    return JSONResponse(body, status_code=200 if srv.ready else 503)


@app.get("/v1/audio/voices")
async def voices() -> dict:
    return {"voices": [{"name": v} for v in sorted(srv.voices)]}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "drama-3", "object": "model"}]}


@app.get("/v1/engine")
async def engine_info() -> dict:
    return {"startup": srv.startup, **srv.counts}


def to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")  # drama3/bench.py の保存と同じ変換
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
    voice = body.voice or VOICE
    if voice not in srv.voices:
        raise HTTPException(404, f"unknown voice {voice!r}; available: {sorted(srv.voices)}")
    from fish_speech.models.text2semantic.errors import InvalidInputError

    started = time.perf_counter()  # server/server.py と同じく、前の合成を待つ時間も X-Synthesis-Seconds に入れる
    async with srv.lock:
        try:
            audio, st = await asyncio.to_thread(srv.synthesize, text, voice, body.seed)
        except InvalidInputError as e:
            raise HTTPException(400, str(e)) from e
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise HTTPException(503, f"out of GPU memory: {e}") from e
    elapsed = time.perf_counter() - started
    srv.counts["requests"] += 1
    srv.counts["prefill_captures"] += st["prefill_captures"]
    duration = len(audio) / SAMPLE_RATE
    if st["limited"] or duration > max_seconds(len(text)):
        srv.counts["frame_limited"] += 1
        raise HTTPException(500, f"implausible duration ({st['frames'] * srv.hop / SAMPLE_RATE:.1f}s of frames) "
                                 f"for {len(text)} chars (frames {st['frames']})")
    log.info("voice=%s chars=%d frames=%d chunks=%d audio=%.2fs synth=%.3fs captures=%d", voice, len(text), st["frames"],
             st["chunks"], duration, elapsed, st["prefill_captures"])
    return Response(to_wav(audio), media_type="audio/wav",
                    headers={"X-Synthesis-Seconds": f"{elapsed:.3f}", "X-Audio-Seconds": f"{duration:.3f}"})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "18098")), log_level="info")
