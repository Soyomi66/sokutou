#!/usr/bin/env bash
# sokutou のイメージを 3 つ作る（docker/Dockerfile の段ごとに 1 つ）。最後に server のタグを出す。
#   scripts/build_voice_engine.sh
#   sokutou/official:7ed4374   公式の実装と音声合成サーバー（速さと出力の基準）
#   sokutou/dev:7ed4374        official + CuPy。ref/ と engine/ のスクリプトを動かす（scripts/run_in_image.sh）
#   sokutou/server:<rev>       dev + engine/。エンジンを差し込んだ音声合成サーバー
# <rev> は git のコミット。server/、engine/、Dockerfile に commit していない変更があれば、
# -dirty-<その中身の短いハッシュ> を付ける。
# ビルドに渡すのは server/*.py、engine/*.py、Dockerfile だけ（runs/ や声のファイルを渡さないため、tar で選ぶ）。
# 初めて作るときは、Irodori-TTS の依存（torch など、約 8GB）を取得する。
set -euo pipefail
cd "$(dirname "$0")/.."
files=(docker/Dockerfile server/*.py engine/*.py)
rev=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- "${files[@]}")" ]; then
  rev="$rev-dirty-$(cat "${files[@]}" | sha256sum | cut -c1-8)"
fi
irodori=7ed4374
build() {  # build <段> <タグ>
  tar -c "${files[@]}" \
    | docker build -f docker/Dockerfile --target "$1" --build-arg "SOKUTOU_REVISION=$rev" -t "$2" - >&2
}
build official "sokutou/official:$irodori"
build dev "sokutou/dev:$irodori"
build server "sokutou/server:$rev"
echo "sokutou/server:$rev"
