"""読み間違いと声の近さを比べるための音声を作り、電子透かしが検出できるかも確かめる。

48 文 × シード 1, 2 を、公式の実装（official）かエンジン（fast-fp32、fast-tf32、fast-bf16 など）で合成し、
音声合成サーバーが返すのと同じ音声（24kHz に変換して前後の無音を切り落としたもの）を WAV で保存する。
作者の前の計測（非公開）の gen_quality.py と同じ流れ（ref/common.py の引数）。

電子透かし（SilentCipher）は、公式の検出（decode_wav）で次の 3 つを調べる。
- 保存した 24kHz の音声（16 ビットにした後。サーバーが返すもの）
- 電子透かしを入れた直後の 48kHz の音声
- 電子透かしを入れる前の 48kHz の音声（対照。検出されないはず）
「検出できた」は、読み出した文字列が Irodori の印（IRDTS = 73, 82, 68, 84, 83）と一致したとき。

  scripts/run_in_image.sh engine/gen_quality.py --cond fast-tf32
  scripts/run_in_image.sh engine/gen_quality.py --cond official-s20        # 公式の実装の 20 ステップ
  IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/gen_quality.py --cond fused-bf16-nowm-bdec-fp8-s20
条件の名前は、official か <エンジン>-<精度>（エンジンは fast・bucket・fused）に、次をこの順に付けられる
（ref/bench_engine.py の出力の名前と同じ並び）:
-nowm（電子透かしを外す。透かしの検出は調べない）、-hdec（復号を float16）、-bdec（復号を決まった長さで動かす）、
-fp8（行列積を FP8 にする。fused だけ。-fp8cfg・-fp8nocfg は CFG あり・なしのステップだけ）、
-mf か -large（モデルの印。IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF か Aratako/Irodori-TTS-v4-Large と一緒に使う）、-s<数>（ステップ数。なければモデルの既定で、RF 版 40、MF 版 4）。
  IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF scripts/run_in_image.sh engine/gen_quality.py --cond official-nowm-mf
出力: runs/quality/wav/<cond>_s<seed>_<文の番号>.wav、runs/quality/rows-<cond>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/ref")
sys.path.insert(0, "/work")
from common import (CHECKPOINT, MODEL_STEPS, OUTPUT_SR, WORK, load_runtime, load_sentences, make_request, normalize_text, set_num_steps,  # noqa: E402
                    timed_synth)
from irodori_tts.watermark import IRODORI_WATERMARK_PAYLOAD  # noqa: E402

PAYLOAD = list(IRODORI_WATERMARK_PAYLOAD)


def save(path: Path, audio: np.ndarray) -> np.ndarray:
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUTPUT_SR)
        w.writeframes(pcm.tobytes())
    return pcm.astype(np.float32) / 32768


def detect(wm_model, audio, sr: int) -> dict:
    y = torch.as_tensor(np.asarray(audio), dtype=torch.float32).reshape(-1)
    res = wm_model.decode_wav(y, int(sr), phase_shift_decoding=True)
    msg = res["messages"][0] if res.get("status") and res.get("messages") else None
    conf = float(res["confidences"][0]) if res.get("status") and res.get("confidences") else None
    return {"ok": msg == PAYLOAD, "confidence": conf, "message": msg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="official か <fast|bucket|fused>-<fp32|tf32|bf16>。後ろに -fp8、-hdec、-s<ステップ数> を順に付けられる")
    ap.add_argument("--seeds", default="1,2")
    args = ap.parse_args()
    m = re.fullmatch(r"(official|(fast|bucket|fused)-(fp32|tf32|bf16))(-nowm)?(-hdec)?(-bdec)?(-fp8(?:cfg|nocfg)?)?(-mf|-large)?(?:-s(\d+))?",
                     args.cond)
    assert m, f"条件の名前が読めない: {args.cond}"
    engine, precision, nowm, hdec, bdec = m.group(2), m.group(3), bool(m.group(4)), bool(m.group(5)), bool(m.group(6))
    fp8 = (m.group(7)[4:] or "all") if m.group(7) else None  # all・cfg・nocfg
    assert (m.group(8) or "") == CHECKPOINT["tag"], f"条件の名前のモデルの印（-mf、-large）と、IRODORI_CHECKPOINT のモデル（印 {CHECKPOINT['tag'] or 'なし'}）が合わない"
    steps = int(m.group(9) or MODEL_STEPS)
    assert not (fp8 and engine != "fused")
    set_num_steps(steps)
    rt = load_runtime()
    assert rt.watermarker.ready, "SilentCipher が読み込めていない"
    wm_model = rt.watermarker.model
    if nowm:
        from engine.bucket import disable_watermark
        disable_watermark(rt)
    sampler = decoder = None
    if hdec:  # 復号を float16 にする（出力が変わる）
        from engine.fused import half_decode
        half_decode(rt)
    if engine in ("bucket", "fused"):  # 起動の準備だけ（48 文の形は使わない）
        from engine.bucket import install_bucket, prewarm_codec
        if engine == "fused":
            from engine.fused import install_fused as install_bucket
        sampler = install_bucket(rt, precision=precision, num_steps=steps, **({"fp8": fp8} if fp8 else {}))
        with torch.inference_mode():
            ref_latent, ref_mask = rt._load_reference_latent(req=make_request("あ。", 1.0, 0), batch_size=1, messages=[])
            print("prewarmed", sampler.prewarm(sampler.voice_for(ref_latent, ref_mask)), prewarm_codec(rt), flush=True)
    elif engine == "fast":
        from engine.fast import install
        sampler = install(rt, precision=precision, num_steps=steps)
    if bdec:  # half_decode の後に入れる（engine/decode.py）
        from engine.decode import install_bucket_decoder
        decoder = install_bucket_decoder(rt)
        print("prewarmed decode", decoder.prewarm(), flush=True)
    got: dict = {}
    wm = rt.watermarker.encode_batch

    def watermark(audios, *, sample_rate):
        out = wm(audios, sample_rate=sample_rate)
        got["pre"] = audios[0].detach().float().cpu().reshape(-1).numpy()
        got["post"] = torch.as_tensor(out[0]).detach().float().cpu().reshape(-1).numpy()
        got["sr"] = int(sample_rate)
        return out

    rt.watermarker.encode_batch = watermark
    sentences = load_sentences()
    for text in ("準備ができました。", "今日もよろしくお願いします。"):
        timed_synth(rt, make_request(text, 2.0, 0))
    out_dir = WORK / "runs" / "quality"
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (int(x) for x in args.seeds.split(",")):
        for i, s in enumerate(sentences):
            text = normalize_text(s["text"])
            got.clear()
            audio, r = timed_synth(rt, make_request(text, s["seconds"], seed))
            name = f"{args.cond}_s{seed}_{i:02d}.wav"
            saved = save(out_dir / "wav" / name, audio)
            rows.append({
                "cond": args.cond, "i": i, "seed": seed, "file": f"wav/{name}", "text": text, "seconds": s["seconds"],
                "audio_sec": round(len(audio) / OUTPUT_SR, 3),
                # 透かしを外した条件では調べない（None）
                "wm_24k": None if nowm else detect(wm_model, saved, OUTPUT_SR),
                "wm_48k": None if nowm else detect(wm_model, got["post"], got["sr"]),
                "wm_48k_before": None if nowm else detect(wm_model, got["pre"], got["sr"]),
            })
        print("seed done", args.cond, seed, flush=True)
    if sampler is not None:
        assert sampler.fallbacks == 0, "公式の関数に落ちた"
        assert getattr(sampler, "late_captures", 0) == 0, "合成の途中でグラフを記録した"
    if decoder is not None:
        assert decoder.cold == 0 and decoder.misses == 0, "復号が、起動のときに通していない長さを使った"
    (out_dir / f"rows-{args.cond}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(rows)
    for k in () if nowm else ("wm_24k", "wm_48k", "wm_48k_before"):
        ok = sum(r[k]["ok"] for r in rows)
        conf = [r[k]["confidence"] for r in rows if r[k]["confidence"] is not None]
        print(f"{args.cond} {k}: 検出 {ok}/{n}  confidence 最小 {min(conf) if conf else float('nan'):.3f} "
              f"中央値 {float(np.median(conf)) if conf else float('nan'):.3f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
