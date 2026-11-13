"""音声合成サーバー（server/server.py）に、このエンジンを差し込んで動かす。

API、入力の確かめ、24kHz への変換、無音の切り落とし、応答のヘッダーは、server/server.py をそのまま使う
（イメージの /app/server.py。docker/Dockerfile の server の段）。ここでは、server.py がモデルを読み込んだあとに、
次の 4 つを足す（ref/bench_engine.py の --engine fused --bucket-decode --no-watermark と同じ順）。
- 電子透かし（SilentCipher）を外す（2026-11-11 に利用者が外してよいと決めた）
- DiT を engine/fused.py（BF16、自作のカーネル、バケットごとの CUDA Graph）に差し替え、声の CUDA Graph を記録する
- 復号を決まった長さで動かし（engine/decode.py）、全部の長さを通しておく
- 動いた回数を数え、/v1/engine で返す

環境変数（server.py のものに加えて）
- IRODORI_NUM_STEPS: 必須。エンジンと要求のステップ数を、この 1 つの値から作る（2026-11-11 に 20 と決めた）。
  エンジンは、要求のステップ数が自分と違うと公式の関数に戻る（engine/fast.py の _supported）ので、空なら起動しない。
- IRODORI_CHECKPOINT（server.py のもの）: MF 版（Aratako/Irodori-TTS-v4.1-Small-MF）と v4-Large
  （Aratako/Irodori-TTS-v4-Large）も使える。MF 版は CFG がなく、ステップ数の既定は 4（IRODORI_NUM_STEPS=4）。
  v4-Large は Small と同じ設定（20 ステップ、CFG）で、GPU のメモリーを約 31GB 使う。
- SOKUTOU_WATERMARK: 1 なら電子透かしを残す（既定 0）
- IRODORI_PRECISION は fp32 のままにする（エンジンが fp32 の重みから BF16 の複製を作る）
- SOKUTOU_OFFICIAL_TAIL: 1 なら、文末を探す関数を公式のままにする（既定 0。engine/tail.py。出力は同じで、遅くなる）
- SOKUTOU_FUSED_DECODE: 0 なら、復号を自作のカーネルにまとめない（既定 1。engine/decode.py。出力は同じで、遅くなる）

起動のときの試し合成（server.py の lifespan）で、公式の関数に戻るか、グラフを途中で記録するか、起動のときに通していない
長さで復号したら、起動を止める。動いているあいだに起きたら、エラーのログを出して数を増やす（音声はそのまま返す）。

  /opt/irodori/.venv/bin/python -m engine.serve    # 作業ディレクトリは engine/ の親（イメージでは /opt/sokutou）
"""

from __future__ import annotations

import logging
import os
import time

import torch

import server  # server/server.py（PYTHONPATH の /app）

from engine.bucket import disable_watermark, prewarm_codec
from engine.decode import install_bucket_decoder
from engine.fused import install_fused

log = logging.getLogger("voice-engine")

KEEP_WATERMARK = os.environ.get("SOKUTOU_WATERMARK", "0") == "1"

state: dict = {"sampler": None, "decoder": None, "startup": {}, "requests": 0, "irregular_requests": 0}


class _Captured(Exception):
    def __init__(self, req) -> None:
        super().__init__("captured")
        self.req = req


class _CaptureRuntime:
    """server.Engine.synthesize が組み立てた SamplingRequest を受け取り、合成せずに投げ返す。"""

    def synthesize(self, req, log_fn=None):  # noqa: ARG002
        raise _Captured(req)


def request_for(engine: server.Engine, voice: str):
    """server.py がこの声に送るのと同じ SamplingRequest（参照音声の渡し方を含む）を作る。"""
    runtime = engine.runtime
    engine.runtime = _CaptureRuntime()
    try:
        _original_synthesize(engine, "あ。", voice, 1.0, 0, 1.0)
    except _Captured as c:
        return c.req
    finally:
        engine.runtime = runtime
    raise RuntimeError("server.Engine.synthesize が runtime.synthesize を呼ばなかった")


def counters() -> dict:
    s, d = state["sampler"], state["decoder"]
    return {"fallbacks": s.fallbacks, "late_captures": s.late_captures, "decode_misses": d.misses, "decode_cold": d.cold}


_original_load = server.Engine.load
_original_synthesize = server.Engine.synthesize


def load(self: server.Engine) -> None:
    if server.NUM_STEPS is None:
        raise RuntimeError("IRODORI_NUM_STEPS を設定する（例: 20）。空だと要求は 40 ステップになる")
    if server.PRECISION != "fp32":
        raise RuntimeError(f"IRODORI_PRECISION は fp32 にする（今は {server.PRECISION}）")
    if len(self.voices) != 1:
        # 声ごとに CUDA Graph とメモリー（約 4GB）が要り、2 つ以上の声での切り替えは確かめていない（TASKS.md）
        raise RuntimeError(f"声は 1 つだけにする（今は {list(self.voices)}）")
    _original_load(self)
    rt = self.runtime
    t0 = time.perf_counter()
    if not KEEP_WATERMARK:
        disable_watermark(rt)
    sampler = install_fused(rt, precision="bf16", num_steps=server.NUM_STEPS)
    info: dict = {"checkpoint": server.CHECKPOINT, "flow": str(rt.model_cfg.flow_parameterization),
                  "num_steps": sampler.num_steps, "cfg_steps": sampler.n_cfg, "watermark": rt.watermarker.ready}
    with torch.inference_mode():
        for name in self.voices:
            ref_latent, ref_mask = rt._load_reference_latent(req=request_for(self, name), batch_size=1, messages=[])
            info.update(sampler.prewarm(sampler.voice_for(ref_latent, ref_mask)))
    info.update(prewarm_codec(rt))
    decoder = install_bucket_decoder(rt)
    info.update(decoder.prewarm())
    free, total = torch.cuda.mem_get_info()
    info.update(engine_startup_sec=round(time.perf_counter() - t0, 3), device_used_gb=round((total - free) / 1e9, 2))
    state.update(sampler=sampler, decoder=decoder, startup=info)
    log.info("engine ready: %s", info)


def synthesize(self: server.Engine, *args, **kwargs):
    before = counters()
    out = _original_synthesize(self, *args, **kwargs)
    after = counters()
    state["requests"] += 1
    if after != before:
        if not self.ready:  # lifespan の試し合成。設定の誤りなので起動を止める
            raise RuntimeError(f"試し合成がエンジンの外を通った: {before} -> {after}")
        state["irregular_requests"] += 1
        log.error("エンジンの外を通った合成: %s -> %s（args=%s）", before, after, args[:1])
    return out


server.Engine.load = load
server.Engine.synthesize = synthesize


@server.app.get("/v1/engine")
async def engine_stats() -> dict:
    """エンジンの設定と、動いた回数。irregular_requests は公式の関数に戻ったなどの合成の数（0 のはず）。"""
    s = state["sampler"]
    return {
        "engine": "sokutou fused-bf16",
        "ready": server.engine.ready,
        "checkpoint": server.CHECKPOINT,
        "num_steps": None if s is None else s.num_steps,
        "requests": state["requests"],
        "irregular_requests": state["irregular_requests"],
        **({} if s is None else counters()),
        "startup": state["startup"],
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(server.app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8088")), log_level="info")
