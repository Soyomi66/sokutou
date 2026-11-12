# sokutou のイメージ。公開されているもの（python のイメージ、GitHub の Irodori-TTS、PyPI）だけから作る。
# 作り方は scripts/build_voice_engine.sh（ビルドに渡すのは server/、engine/、この Dockerfile だけ）。
# 段は 3 つで、前の段の上に重ねる。
#   official: 公式の実装（Irodori-TTS の commit 7ed4374、torch 2.10 + CUDA 12.8、fp32）と、
#             それを包む音声合成サーバー（server/server.py）。公式の実装の速さと出力を測る基準
#   dev:      official に CuPy（自作の CUDA カーネルを NVRTC でコンパイルする）と C コンパイラーを足したもの。
#             ref/ と engine/ のスクリプトを scripts/run_in_image.sh で動かす
#   server:   dev に engine/ を入れ、エンジンを差し込んだ音声合成サーバー（engine/serve.py）を動かす
# モデルの重みはイメージに入れない。Hugging Face のキャッシュをつないで使う（README の「動かし方」）。

FROM python:3.10-slim-bookworm AS official

RUN pip install --no-cache-dir uv==0.10.9

# git: 依存の silentcipher を git から入れるため。ffmpeg: torchaudio が音声の読み込みに使う。
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# エンジンは、この版の公式の実装と数値で突き合わせてある。版を変えるときは、突き合わせからやり直す。
ARG IRODORI_COMMIT=7ed4374eaf49355279db51307e8141984f900836
RUN git clone https://github.com/Aratako/Irodori-TTS.git /opt/irodori \
    && git -C /opt/irodori checkout ${IRODORI_COMMIT}

WORKDIR /opt/irodori
# UV_NO_CACHE: ダウンロードしたパッケージのキャッシュ（約 8GB）をイメージに残さない。
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
RUN uv sync --frozen --extra cu128 --no-dev \
    && uv pip install --python .venv/bin/python fastapi==0.129.2 "uvicorn[standard]==0.41.0"

COPY server/server.py server/trim.py /app/
ENV PYTHONPATH=/opt/irodori:/app \
    HF_HUB_OFFLINE=1 \
    VOICE_DIR=/voices \
    PORT=8088
EXPOSE 8088
CMD ["/opt/irodori/.venv/bin/python", "/app/server.py"]


FROM official AS dev

# gcc: torch.compile を試すときに要る（エンジンは使わない）。
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# torch・numpy などの版を変えないよう、cupy は依存を解決せずに入れる（--no-deps）。
RUN uv pip install --python /opt/irodori/.venv/bin/python --no-deps cupy-cuda12x==13.6.0 fastrlock==0.8.3


FROM dev AS server

COPY engine/ /opt/sokutou/engine/
ARG SOKUTOU_REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/kosuketut/sokutou \
      org.opencontainers.image.revision=${SOKUTOU_REVISION}
ENV PYTHONPATH=/opt/sokutou:/opt/irodori:/app \
    SOKUTOU_REVISION=${SOKUTOU_REVISION}
WORKDIR /opt/sokutou
CMD ["/opt/irodori/.venv/bin/python", "-m", "engine.serve"]
