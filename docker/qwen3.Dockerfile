# Qwen3-TTS（12Hz、CustomVoice）の公式の実装のイメージ。公開されているもの（python のイメージ、PyPI、
# PyTorch の配布元）だけから作る。Irodori-TTS のイメージ（docker/Dockerfile）とは依存の版が違うので分ける
# （qwen-tts 0.1.1 は transformers 4.57.3 に固定している）。
#   docker build --target qwen3-official -t sokutou/qwen3-official:0.1.1 - < docker/qwen3.Dockerfile
# モデルの重みはイメージに入れない。Hugging Face のキャッシュをつないで使う。
# 段
#   qwen3-official: 公式の実装（qwen-tts 0.1.1、torch 2.10 + CUDA 12.8）。速さと出力の基準
#   qwen3-dev:      qwen3-official に gcc と CuPy を足したもの（エンジン。torch.compile の Triton が gcc を使う）
#     docker build --target qwen3-dev -t sokutou/qwen3-dev:0.1.1 - < docker/qwen3.Dockerfile
#   qwen3-server:   qwen3-dev に qwen3/ のエンジンを入れ、音声合成サーバー（qwen3/serve.py）を動かす。
#     qwen3/ を使うので、scripts/build_qwen3.sh で作る（qwen3/*.py とこの Dockerfile だけを tar で渡す）

FROM python:3.12-slim-bookworm AS qwen3-official

RUN pip install --no-cache-dir uv==0.10.9
# sox: qwen-tts が import する sox（Python）が呼ぶ。ffmpeg: torchaudio と librosa が音声を読むときに使う。
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
# torch と torchaudio は CUDA 12.8 の版を PyTorch の配布元から取る。ほかは PyPI。
RUN uv venv /opt/qwen3/.venv --python /usr/local/bin/python3.12 \
    && uv pip install --python /opt/qwen3/.venv/bin/python \
         --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
         --index-strategy unsafe-best-match \
         torch==2.10.0 torchaudio==2.10.0 qwen-tts==0.1.1 \
    && /opt/qwen3/.venv/bin/python -c "import qwen_tts, torch, transformers; print(torch.__version__, transformers.__version__)"

ENV HF_HUB_OFFLINE=1
WORKDIR /work


FROM qwen3-official AS qwen3-dev

# Triton（torch.compile の inductor が使う）は、最初に動くときに小さな C のモジュールを gcc で作る。
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# torch・numpy などの版を変えないよう、cupy は依存を解決せずに入れる（--no-deps）。
RUN uv pip install --python /opt/qwen3/.venv/bin/python --no-deps cupy-cuda12x==13.6.0 fastrlock==0.8.3


FROM qwen3-dev AS qwen3-server

COPY qwen3/ /opt/sokutou/qwen3/
ARG SOKUTOU_REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/kosuketut/sokutou \
      org.opencontainers.image.revision=${SOKUTOU_REVISION}
ENV SOKUTOU_REVISION=${SOKUTOU_REVISION} PORT=18096
WORKDIR /opt/sokutou
EXPOSE 18096
CMD ["/opt/qwen3/.venv/bin/python", "/opt/sokutou/qwen3/serve.py"]
