#!/usr/bin/env bash
# イメージの中で、このリポジトリの Python スクリプトを GPU で動かす（リポジトリは /work、声は /voices につなぐ）。
#   scripts/run_in_image.sh ref/profile_official.py [引数...]
#   IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh ref/bench_engine.py --engine fused ...   # engine/ を使うとき
# 環境変数
# - IMAGE: 既定 sokutou/official:7ed4374（公式の実装）。engine/fused.py を使うスクリプトは sokutou/dev:7ed4374
# - GPU: 使う GPU の番号（nvidia-smi の番号。既定 0）。その GPU で動いているプロセスがあれば止まる
# - NO_GPU=1: GPU をつながない（CPU で動かす比較用）
# - VOICES: 声のディレクトリ（<声の名前>/*.wav。既定はリポジトリの voices/。git に入れない）
# - HF_CACHE: Hugging Face のキャッシュ（既定 ~/.cache/huggingface）。読み取り専用でつなぎ、取得はしない
# - IRODORI_CHECKPOINT: モデル（既定 Aratako/Irodori-TTS-v4.1-Small。MF 版は Aratako/Irodori-TTS-v4.1-Small-MF）
# - DOCKER_EXTRA: docker run に足す引数
# - PYTHON: イメージの中の Python（既定 /opt/irodori/.venv/bin/python。Qwen3-TTS のイメージは /opt/qwen3/.venv/bin/python）
# 自分の uid で動かすので、runs/ の出力は root の持ち物にならない。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-sokutou/official:7ed4374}
GPU=${GPU:-0}
VOICES=$(readlink -f "${VOICES:-$REPO/voices}")
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
[ $# -ge 1 ] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }
[ -d "$VOICES" ] || { echo "声のディレクトリがない: $VOICES（VOICES で指定する）" >&2; exit 2; }

gpu_args=(--gpus "\"device=$GPU\"")
if [ -n "${NO_GPU:-}" ]; then
  gpu_args=()
else
  busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader)
  if [ -n "$busy" ]; then
    echo "GPU $GPU で動いているプロセスがある。止めてから実行する:" >&2
    echo "$busy" >&2
    exit 1
  fi
fi
mkdir -p "$REPO/runs"
exec docker run --rm "${gpu_args[@]}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e USER=bench -e TORCHINDUCTOR_CACHE_DIR=/tmp/home/inductor \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  ${REF_WAV:+-e REF_WAV="$REF_WAV"} \
  ${IRODORI_CHECKPOINT:+-e IRODORI_CHECKPOINT="$IRODORI_CHECKPOINT"} \
  -v "$HF_CACHE:/hf:ro" \
  -v "$VOICES:/voices:ro" \
  -v "$REPO:/work" -w /work \
  ${DOCKER_EXTRA:-} \
  "$IMAGE" "${PYTHON:-/opt/irodori/.venv/bin/python}" "$@"
