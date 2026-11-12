"""公式の実装（イメージの /opt/irodori）を、音声合成サーバー（server/server.py）と同じ設定で動かすための共通部分。

server.py と同じ RuntimeKey と SamplingRequest を組み立てる。イメージの中で動かす（scripts/run_in_image.sh）。
参照音声は、環境変数 REF_WAV（イメージの中のパス）か、/voices の名前順で最初の声の最初の WAV。
モデルは、環境変数 IRODORI_CHECKPOINT（サーバーと同じ名前。既定 Aratako/Irodori-TTS-v4.1-Small）。
使えるモデルは CHECKPOINTS の表にあるものだけ。MF 版（MeanFlow）を選ぶと、記録の名前に -mf が付き、ステップ数の既定が 4 になる。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, "/app")
from trim import trim_audio  # noqa: E402

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    resolve_cfg_scales,
)

WORK = Path("/work")
VOICES = Path("/voices")


def _default_ref_wav() -> str:
    """server.py の既定の声（名前順で最初の声）の、最初の WAV。"""
    dirs = sorted(p for p in VOICES.iterdir() if p.is_dir()) if VOICES.is_dir() else []
    for d in dirs:
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            return str(wavs[0])
    return str(VOICES / "voice" / "ref.wav")  # 声がないときは読み込みで失敗する


REF_WAV = os.environ.get("REF_WAV") or _default_ref_wav()
OUTPUT_SR = 24000
RF_REPO = "Aratako/Irodori-TTS-v4.1-Small"
# 確かめたモデル。tag は記録の名前に付ける印（同じ名前の記録を別のモデルで上書きしないため）、model_steps はモデルの既定の
# ステップ数（公式の runtime が、要求の num_steps が None のときに使う値）、default_steps は計測と確かめの --steps の既定、
# flow はモデルの設定の flow_parameterization（読み込んだあとで合うかを確かめる）。表にないモデルは動かさない。
CHECKPOINTS = {
    RF_REPO: {"tag": "", "model_steps": 40, "default_steps": 20, "flow": "rf_velocity"},
    # MeanFlow で蒸留した版。CFG がなく、既定 4 ステップ。
    "Aratako/Irodori-TTS-v4.1-Small-MF": {"tag": "-mf", "model_steps": 4, "default_steps": 4, "flow": "meanflow"},
    # 3.3B。文と caption のエンコーダーは T5Gemma 2（重みは model.safetensors に入っている）。Gemma の利用規約。
    # ステップ数と CFG は Small と同じ設定から始める（20 ステップ、CFG は前半 10 ステップ）。
    "Aratako/Irodori-TTS-v4-Large": {"tag": "-large", "model_steps": 40, "default_steps": 20, "flow": "rf_velocity"},
}
CHECKPOINT_REPO = os.environ.get("IRODORI_CHECKPOINT") or RF_REPO
if CHECKPOINT_REPO not in CHECKPOINTS:
    raise RuntimeError(f"IRODORI_CHECKPOINT={CHECKPOINT_REPO} は ref/common.py の CHECKPOINTS にない（確かめたのは {list(CHECKPOINTS)}）")
CHECKPOINT = CHECKPOINTS[CHECKPOINT_REPO]
MEANFLOW = CHECKPOINT["flow"] == "meanflow"
MODEL_STEPS = CHECKPOINT["model_steps"]
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
TERMINATORS = ("。", "！", "？", "!", "?", "…", "♪", "〜", "ー", "」", "』", ")", "）")

TEXT_SCALE, CAPTION_SCALE, SPEAKER_SCALE, _ = resolve_cfg_scales(
    cfg_guidance_mode="independent",
    cfg_scale_text=3.0,
    cfg_scale_caption=3.0,
    cfg_scale_speaker=5.0,
    cfg_scale=None,
    use_caption_condition=False,
    use_speaker_condition=True,
)


# make_request が使うステップ数の既定。None ならモデルの既定（RF 版は 40、MF 版は 4）。音声合成サーバーは環境変数
# IRODORI_NUM_STEPS で変える。計測のスクリプトは --steps を set_num_steps に渡す。
NUM_STEPS: int | None = None

# 計測と確かめのスクリプト（bench_engine.py、engine/check.py、engine/edges.py）の --steps の既定。
# 2026-11-11 に 20 ステップに決めた（CFG は前半 10 ステップ）。モデルの既定は 40 ステップ。
# エンジンは要求の num_steps が自分のステップ数と違うと、黙って公式の関数に戻る（engine/fast.py の _supported）。
# エンジンと要求のステップ数は、同じ値から作る。MF 版はモデルの既定の 4（CHECKPOINTS の default_steps）。
DEFAULT_STEPS = CHECKPOINT["default_steps"]


def set_num_steps(n: int | None) -> None:
    """このプロセスの make_request のステップ数を決める。モデルの既定（RF 版 40、MF 版 4）のときは None のままにする。"""
    global NUM_STEPS
    NUM_STEPS = None if n is None or int(n) == MODEL_STEPS else int(n)


def steps_tag(n: int | None) -> str:
    """出力のファイル名に付ける印。モデルの印（MF 版なら -mf）に、モデルの既定と違うステップ数なら -s<数> を足す。RF 版の 40 ステップなら空。"""
    return CHECKPOINT["tag"] + ("" if n is None or int(n) == MODEL_STEPS else f"-s{int(n)}")


def normalize_text(text: str) -> str:  # server.py と同じ
    text = " ".join(text.split())
    if text and not text.endswith(TERMINATORS):
        text += "。"
    return text


def load_sentences() -> list[dict]:
    return json.loads((WORK / "data" / "sentences.json").read_text(encoding="utf-8"))


def load_runtime(device: str = "cuda", *, compile_model: bool = False, repo: str = CHECKPOINT_REPO) -> InferenceRuntime:
    rt = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint(repo),
            model_device=device,
            codec_repo=CODEC_REPO,
            model_precision="fp32",
            codec_device=device,
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=compile_model,
            compile_dynamic=False,
            split_rare_tokens=True,
        )
    )
    flow = str(rt.model_cfg.flow_parameterization)
    if repo in CHECKPOINTS and flow != CHECKPOINTS[repo]["flow"]:
        raise RuntimeError(f"{repo} の flow_parameterization は {flow}（ref/common.py の CHECKPOINTS では {CHECKPOINTS[repo]['flow']}）")
    return rt


def make_request(text: str, seconds: float | None, seed: int, *, ref_wav: str | None = REF_WAV,
                 num_steps: int | None = None) -> SamplingRequest:
    """server.py の Engine.synthesize と同じ引数（speed=1.0）。"""
    return SamplingRequest(
        text=text, caption=None,
        ref_wav=ref_wav, ref_wavs=None, ref_latent=None, ref_latents=None, ref_embed=None, no_ref=False,
        ref_normalize_db=-16.0, ref_ensure_max=True, num_candidates=1, decode_mode="sequential",
        seconds=seconds, duration_scale=1.0, max_ref_seconds=None, max_text_len=None, max_caption_len=None,
        num_steps=NUM_STEPS if num_steps is None else num_steps, cfg_scale_text=TEXT_SCALE, cfg_scale_caption=CAPTION_SCALE,
        cfg_scale_speaker=SPEAKER_SCALE,
        cfg_guidance_mode="independent", cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
        truncation_factor=None, rescale_k=None, rescale_sigma=None, context_kv_cache=True,
        speaker_kv_scale=None, speaker_kv_min_t=None, speaker_kv_max_layers=None, speaker_uncond_mode="mask",
        seed=seed, t_schedule_mode="linear", sway_coeff=-1.0, trim_tail=True, tail_window_size=20,
        tail_std_threshold=0.05, tail_mean_threshold=0.1, lora_adapter=None,
    )


def postprocess(result) -> np.ndarray:
    """server.py と同じく、24kHz に変換して前後の無音を切り落とす。"""
    audio = result.audio
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu()
    audio = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
    sr = int(result.sample_rate)
    if sr != OUTPUT_SR:
        audio = torchaudio.functional.resample(audio, sr, OUTPUT_SR)
    return np.clip(trim_audio(audio.reshape(-1).numpy(), OUTPUT_SR), -1.0, 1.0)


def timed_synth(rt: InferenceRuntime, req: SamplingRequest) -> tuple[np.ndarray, dict]:
    """bench_tts.py と同じ範囲（合成 → 24kHz → 無音の切り落とし）を測る。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = rt.synthesize(req, log_fn=None)
    out = postprocess(result)
    total = time.perf_counter() - t0
    stages: dict[str, float] = {}
    for name, sec in result.stage_timings:
        stages[name] = stages.get(name, 0.0) + float(sec)
    return out, {"total_sec": total, "stages": stages, "audio_sec": len(out) / OUTPUT_SR}


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "med": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}
