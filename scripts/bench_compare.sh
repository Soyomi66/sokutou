#!/usr/bin/env bash
# 冷えた状態の速さを、条件を交互に、1 つずつ順に測る（前のコンテナが終わってから次を始める）。
#   scripts/bench_compare.sh <runs/ の下の出力先> <回数> <条件>...
# 条件は "<イメージの種類>:<bench_engine.py の引数>"。イメージの種類は base（sokutou/official）か cc（sokutou/dev。cupy 入り）。
# 例: scripts/bench_compare.sh bench/fused 2 "cc:--engine bucket" "cc:--engine fused"
# どの条件にも --precision bf16 --no-watermark を付ける。出力のファイル名の後ろに -<イメージの種類>-<回> が付く。
# ステップ数の既定は 20（ref/common.py の DEFAULT_STEPS）。40 ステップの前の記録と比べる条件には --steps 40 を付ける。
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1; rounds=$2; shift 2
mkdir -p "runs/$out"
for r in $(seq 1 "$rounds"); do
  for cond in "$@"; do
    kind=${cond%%:*}; args=${cond#*:}
    case $kind in
      base) image=sokutou/official:7ed4374 ;;
      cc) image=sokutou/dev:7ed4374 ;;
      *) echo "イメージの種類は base か cc: $kind" >&2; exit 2 ;;
    esac
    # 前のコンテナの GPU のメモリーが解放されるまで待つ
    for _ in $(seq 1 30); do
      [ -z "$(nvidia-smi -i "${GPU:-0}" --query-compute-apps=pid --format=csv,noheader)" ] && break
      sleep 1
    done
    echo "== round $r: $kind: $args" >&2
    # shellcheck disable=SC2086
    IMAGE=$image scripts/run_in_image.sh ref/bench_engine.py $args --precision bf16 --no-watermark \
      --out-dir "$out" "--suffix=-$kind-$r" 2>&1 | grep -E "^(prewarmed|end|ALL DONE)|rror|Traceback|assert" >&2 || true
  done
done
python3 tools/bench_summary.py "runs/$out"/*.jsonl
