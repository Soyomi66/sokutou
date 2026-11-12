"""参照音声の声で話す音声合成サーバー（Irodori-TTS v4.1-Small を包む）。

OpenAI 互換の `POST /v1/audio/speech` で WAV（PCM16 モノラル）を返す。
公式の実装のまま合成する。sokutou のエンジンは engine/serve.py がこのファイルを読み込んで差し込む。
作者の別のリポジトリ（非公開）の音声合成サーバーから写した。写したときに変えたのは、
このファイルの説明の文（この段落と、下の seconds の段落）だけ。

声は `VOICE_DIR/<声の名前>/*.wav` に置く。1 つのディレクトリに複数の WAV があれば、
名前順につないで参照音声にする（合計 120 秒まで）。参照音声の書き起こしは要らない。

環境変数
- VOICE_DIR: 声のディレクトリ（既定 /voices）
- DEFAULT_VOICE: `voice` を省いたときの声（既定は名前順で最初の声）
- IRODORI_CHECKPOINT: HF のリポジトリ名かローカルの .safetensors（既定 Aratako/Irodori-TTS-v4.1-Small）
- IRODORI_CODEC_REPO: コーデック（既定 Aratako/Semantic-DACVAE-Japanese-32dim）
- IRODORI_NUM_STEPS: サンプリングのステップ数（空ならモデルの既定。減らすと速くなり、音質が落ちることがある）
- IRODORI_PRECISION: fp32 か bf16（既定 fp32。手元の計測では bf16 でも速くならなかった）
- OUTPUT_SAMPLE_RATE: 出力のサンプリングレート（既定 24000。モデルの出力は 48kHz）
- MAX_CHARS: 1 リクエストの最大文字数（既定 200）
- EMPTY_CACHE_EVERY: この回数の合成ごとに GPU のキャッシュを解放する（既定 50）
- HOST / PORT: 待ち受け（既定 0.0.0.0:8088）

リクエストの `seconds` で話す長さを決められる。モデルの予測は長めで、余った時間に余計な声が入ることがあるため、
呼び出す側で文字数から見積もった長さを送るとよい。
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    resolve_cfg_scales,
)
from trim import trim_audio

log = logging.getLogger("voice-server")

VOICE_DIR = Path(os.environ.get("VOICE_DIR", "/voices"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "")
CHECKPOINT = os.environ.get("IRODORI_CHECKPOINT", "Aratako/Irodori-TTS-v4.1-Small")
CODEC_REPO = os.environ.get("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")
NUM_STEPS = int(os.environ["IRODORI_NUM_STEPS"]) if os.environ.get("IRODORI_NUM_STEPS") else None
PRECISION = os.environ.get("IRODORI_PRECISION", "fp32")
OUTPUT_SR = int(os.environ.get("OUTPUT_SAMPLE_RATE", "24000"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "200"))
EMPTY_CACHE_EVERY = int(os.environ.get("EMPTY_CACHE_EVERY", "50"))

# 文末がこれらで終わらない文には「。」を足す（終わりが分からないと語尾が崩れやすい）。
TERMINATORS = ("。", "！", "？", "!", "?", "…", "♪", "〜", "ー", "」", "』", ")", "）")


class SpeechIn(BaseModel):
    input: str
    model: str = "irodori-tts"
    voice: str = ""
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: int | None = None
    # 話す長さ（秒）。省けばモデルが予測する。speech.cpp と同じく、speed で割った長さで話す。
    seconds: float | None = Field(default=None, ge=0.5, le=30.0)


class Engine:
    """モデルを 1 つ読み込み、合成を 1 件ずつ実行する。"""

    def __init__(self) -> None:
        self.runtime: InferenceRuntime | None = None
        self.voices: dict[str, list[str]] = {}
        self.ready = False
        self.count = 0
        self.lock = asyncio.Lock()
        self.cfg = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=True,
        )

    def scan_voices(self) -> None:
        voices = {}
        for d in sorted(p for p in VOICE_DIR.iterdir() if p.is_dir()):
            wavs = sorted(str(w) for w in d.glob("*.wav"))
            if wavs:
                voices[d.name] = wavs
        if not voices:
            raise RuntimeError(f"no voices found under {VOICE_DIR} (expected <name>/*.wav)")
        self.voices = voices

    def load(self) -> None:
        checkpoint = CHECKPOINT if CHECKPOINT.endswith((".safetensors", ".pt")) else download_hf_checkpoint(CHECKPOINT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint,
                model_device=device,
                codec_repo=CODEC_REPO,
                model_precision=PRECISION,
                codec_device=device,
                codec_precision="fp32",
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
                split_rare_tokens=True,
            )
        )

    def default_voice(self) -> str:
        return DEFAULT_VOICE if DEFAULT_VOICE in self.voices else next(iter(self.voices))

    def synthesize(
        self, text: str, voice: str, speed: float, seed: int | None, seconds: float | None = None
    ) -> tuple[np.ndarray, int]:
        assert self.runtime is not None
        refs = self.voices[voice]
        text_scale, caption_scale, speaker_scale, _ = self.cfg
        result = self.runtime.synthesize(
            SamplingRequest(
                text=text,
                caption=None,
                ref_wav=refs[0] if len(refs) == 1 else None,
                ref_wavs=refs if len(refs) > 1 else None,
                ref_latent=None,
                ref_latents=None,
                ref_embed=None,
                no_ref=False,
                ref_normalize_db=-16.0,
                ref_ensure_max=True,
                num_candidates=1,
                decode_mode="sequential",
                seconds=None if seconds is None else seconds / speed,
                duration_scale=1.0 / speed,
                max_ref_seconds=None,
                max_text_len=None,
                max_caption_len=None,
                num_steps=NUM_STEPS,
                cfg_scale_text=text_scale,
                cfg_scale_caption=caption_scale,
                cfg_scale_speaker=speaker_scale,
                cfg_guidance_mode="independent",
                cfg_scale=None,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                truncation_factor=None,
                rescale_k=None,
                rescale_sigma=None,
                context_kv_cache=True,
                speaker_kv_scale=None,
                speaker_kv_min_t=None,
                speaker_kv_max_layers=None,
                speaker_uncond_mode="mask",
                seed=seed,
                t_schedule_mode="linear",
                sway_coeff=-1.0,
                trim_tail=True,
                tail_window_size=20,
                tail_std_threshold=0.05,
                tail_mean_threshold=0.1,
                lora_adapter=None,
            ),
            log_fn=None,
        )
        audio = result.audio
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu()
        audio = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
        sr = int(result.sample_rate)
        if sr != OUTPUT_SR:
            audio = torchaudio.functional.resample(audio, sr, OUTPUT_SR)
            sr = OUTPUT_SR
        out = trim_audio(audio.reshape(-1).numpy(), sr)
        self.count += 1
        if EMPTY_CACHE_EVERY > 0 and self.count % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.clip(out, -1.0, 1.0), sr


def to_wav(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def normalize_text(text: str) -> str:
    text = " ".join(text.split())
    if text and not text.endswith(TERMINATORS):
        text += "。"
    return text


engine = Engine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.scan_voices()
    log.info("voices: %s", {k: len(v) for k, v in engine.voices.items()})
    await asyncio.to_thread(engine.load)
    # 声ごとに 1 回合成しておく（参照音声の読み込みと GPU の初期化を済ませる）。準備ができるまで /health は 503。
    for name in engine.voices:
        started = time.perf_counter()
        await asyncio.to_thread(engine.synthesize, "準備ができました。", name, 1.0, 0)
        log.info("warmed up voice %s in %.2fs", name, time.perf_counter() - started)
    engine.ready = True
    yield


app = FastAPI(title="voice-server", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    body = {"ok": engine.ready, "voices": list(engine.voices), "default_voice": engine.default_voice() if engine.voices else None}
    return JSONResponse(body, status_code=200 if engine.ready else 503)


@app.get("/v1/audio/voices")
async def voices() -> dict:
    return {"voices": [{"name": k, "references": len(v)} for k, v in engine.voices.items()]}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "irodori-tts", "object": "model"}]}


@app.post("/v1/audio/speech")
async def speech(body: SpeechIn) -> Response:
    if not engine.ready:
        raise HTTPException(503, "model is loading")
    if body.response_format != "wav":
        raise HTTPException(400, "only response_format=wav is supported")
    text = normalize_text(body.input)
    if not text:
        raise HTTPException(400, "input is empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"input is longer than {MAX_CHARS} characters")
    voice = body.voice or engine.default_voice()
    if voice not in engine.voices:
        raise HTTPException(404, f"unknown voice {voice!r}; available: {list(engine.voices)}")
    started = time.perf_counter()
    async with engine.lock:
        try:
            audio, sr = await asyncio.to_thread(engine.synthesize, text, voice, body.speed, body.seed, body.seconds)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise HTTPException(503, f"out of GPU memory: {e}") from e
    elapsed = time.perf_counter() - started
    duration = len(audio) / sr
    # 文字数に比べて長すぎる音声は失敗として返し、バックエンド側で次のエンジンに回す。
    if duration > 2.0 + 0.4 * len(text):
        raise HTTPException(500, f"implausible duration {duration:.1f}s for {len(text)} chars")
    log.info("voice=%s chars=%d audio=%.2fs synth=%.2fs", voice, len(text), duration, elapsed)
    return Response(
        to_wav(audio, sr),
        media_type="audio/wav",
        headers={"X-Synthesis-Seconds": f"{elapsed:.3f}", "X-Audio-Seconds": f"{duration:.3f}"},
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8088")), log_level="info")
