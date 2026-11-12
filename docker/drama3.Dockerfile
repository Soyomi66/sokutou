# Fish Audio Drama 3（fishaudio/drama-3）の公式の実装のイメージ。公開されているもの（python のイメージ、
# GitHub の fish-speech、PyPI、PyTorch の配布元）だけから作る。Irodori-TTS（docker/Dockerfile）と
# Qwen3-TTS（docker/qwen3.Dockerfile）とは依存の版が違うので分ける（fish-speech は torch 2.8.0 と
# transformers 5.19.0 に固定している）。
#   docker build --target drama3-official -t sokutou/drama3-official:0f8a12e - < docker/drama3.Dockerfile
# fish-speech と drama-3 の重みは Fish Audio Community License 1.0 のもの。イメージに重みは入れない。
# Hugging Face のキャッシュをつないで使う。
# 段
#   drama3-official: 公式の実装（fish-speech 0f8a12e、torch 2.8.0 + CUDA 12.8）。速さと出力の基準
#   drama3-dev:      drama3-official に CuPy を足したもの（自作のカーネルのエンジン）
#     docker build --target drama3-dev -t sokutou/drama3-dev:0f8a12e - < docker/drama3.Dockerfile
#   drama3-server:   drama3-dev に FastAPI と drama3/ のエンジンを入れ、音声合成サーバー（drama3/serve.py）を動かす。
#     drama3/ を使うので、scripts/build_drama3.sh で作る（drama3/*.py とこの Dockerfile だけを tar で渡す）

FROM python:3.12-slim-bookworm AS drama3-official

RUN pip install --no-cache-dir uv==0.10.9
# git: fish-speech を取得する。build-essential・portaudio19-dev・libasound-dev: fish-speech が依存する pyaudio を
# 組み立てる（Triton も、最初に動くときに gcc で小さな C のモジュールを作る）。libsox-dev・ffmpeg: 音声の読み書き。
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
         portaudio19-dev libasound-dev libsox-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG FISH_SPEECH_COMMIT=0f8a12e8a651062fdee9c33a894bdccd3a8513f1
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 UV_PYTHON=/usr/local/bin/python3.12
# fish-speech の uv.lock のとおりに入れる（CUDA 12.8 の torch）。
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech \
    && git -C /opt/fish-speech checkout "$FISH_SPEECH_COMMIT" \
    && cd /opt/fish-speech \
    && uv sync --extra cu128 --frozen --no-dev \
    && /opt/fish-speech/.venv/bin/python -c "import fish_speech, torch, transformers, triton; print(torch.__version__, transformers.__version__, triton.__version__)"

ENV HF_HUB_OFFLINE=1 UV_NO_SYNC=1
WORKDIR /work


FROM drama3-official AS drama3-dev

# torch・numpy などの版を変えないよう、cupy は依存を解決せずに入れる（--no-deps）。
RUN uv pip install --python /opt/fish-speech/.venv/bin/python --no-deps cupy-cuda12x==13.6.0 fastrlock==0.8.3


FROM drama3-dev AS drama3-server

# FastAPI（サーバー）。uvicorn と pydantic は fish-speech の依存にあるので、その版のまま使う。
RUN uv pip install --python /opt/fish-speech/.venv/bin/python fastapi==0.129.2 \
    && /opt/fish-speech/.venv/bin/python -c "import fastapi, starlette, pydantic, uvicorn, torch; print(fastapi.__version__, starlette.__version__, pydantic.__version__, uvicorn.__version__, torch.__version__)"
COPY drama3/ /opt/sokutou/drama3/
ARG SOKUTOU_REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/kosuketut/sokutou \
      org.opencontainers.image.revision=${SOKUTOU_REVISION}
ENV SOKUTOU_REVISION=${SOKUTOU_REVISION} PORT=18098
WORKDIR /opt/sokutou
EXPOSE 18098
CMD ["/opt/fish-speech/.venv/bin/python", "/opt/sokutou/drama3/serve.py"]
