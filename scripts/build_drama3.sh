#!/usr/bin/env bash
# Fish Audio Drama 3 のイメージを 3 つ作る（docker/drama3.Dockerfile の段ごとに 1 つ）。最後に drama3-server のタグを出す。
#   scripts/build_drama3.sh
#   sokutou/drama3-official:0f8a12e   公式の実装（fish-speech 0f8a12e）。速さと出力の基準
#   sokutou/drama3-dev:0f8a12e        drama3-official + CuPy。drama3/ のスクリプトを動かす（scripts/run_in_image.sh）
#   sokutou/drama3-server:<rev>       drama3-dev + FastAPI + drama3/。自作のカーネルの音声合成サーバー（drama3/serve.py）
# <rev> は git のコミット。drama3/ と Dockerfile に commit していない変更があれば、-dirty-<その中身の短いハッシュ> を付ける。
# ビルドに渡すのは drama3/*.py と docker/drama3.Dockerfile だけ（runs/ や声のファイルを渡さないため、tar で選ぶ）。
# 重み（41.5GB）はイメージに入れない。Hugging Face のキャッシュをつないで使う。
set -euo pipefail
cd "$(dirname "$0")/.."
files=(docker/drama3.Dockerfile drama3/*.py)
rev=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- "${files[@]}")" ]; then
  rev="$rev-dirty-$(cat "${files[@]}" | sha256sum | cut -c1-8)"
fi
fish=0f8a12e
build() {  # build <段> <タグ>
  tar -c "${files[@]}" \
    | docker build -f docker/drama3.Dockerfile --target "$1" --build-arg "SOKUTOU_REVISION=$rev" -t "$2" - >&2
}
build drama3-official "sokutou/drama3-official:$fish"
build drama3-dev "sokutou/drama3-dev:$fish"
build drama3-server "sokutou/drama3-server:$rev"
echo "sokutou/drama3-server:$rev"
