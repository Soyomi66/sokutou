#!/usr/bin/env bash
# Qwen3-TTS のイメージを 3 つ作る（docker/qwen3.Dockerfile の段ごとに 1 つ）。最後に qwen3-server のタグを出す。
#   scripts/build_qwen3.sh
#   sokutou/qwen3-official:0.1.1   公式の実装（qwen-tts 0.1.1）。速さと出力の基準
#   sokutou/qwen3-dev:0.1.1        qwen3-official + gcc + CuPy。qwen3/ のスクリプトを動かす（scripts/run_in_image.sh）
#   sokutou/qwen3-server:<rev>     qwen3-dev + qwen3/。自作のカーネルの音声合成サーバー（qwen3/serve.py）
# <rev> は git のコミット。qwen3/ と Dockerfile に commit していない変更があれば、-dirty-<その中身の短いハッシュ> を付ける。
# ビルドに渡すのは qwen3/*.py と docker/qwen3.Dockerfile だけ（runs/ や声のファイルを渡さないため、tar で選ぶ）。
# 初めて作るときは、qwen-tts の依存（torch など、約 8GB）を取得する。
set -euo pipefail
cd "$(dirname "$0")/.."
files=(docker/qwen3.Dockerfile qwen3/*.py)
rev=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- "${files[@]}")" ]; then
  rev="$rev-dirty-$(cat "${files[@]}" | sha256sum | cut -c1-8)"
fi
qwen_tts=0.1.1
build() {  # build <段> <タグ>
  tar -c "${files[@]}" \
    | docker build -f docker/qwen3.Dockerfile --target "$1" --build-arg "SOKUTOU_REVISION=$rev" -t "$2" - >&2
}
build qwen3-official "sokutou/qwen3-official:$qwen_tts"
build qwen3-dev "sokutou/qwen3-dev:$qwen_tts"
build qwen3-server "sokutou/qwen3-server:$rev"
echo "sokutou/qwen3-server:$rev"
