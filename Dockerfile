# ベースイメージをCUDA 12.0に変更。CUDA 12.1 だと GPU を認識しなかった
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# CUDA関連の環境変数の設定
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_DISABLE_REQUIRE=true

# pipのタイムアウト設定とリトライ設定
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=5
ENV PIP_TRUSTED_HOST=download.pytorch.org

# システム依存ツールと Git LFS
RUN apt-get update && apt-get install -y \
    python3 python3-pip gcc wget curl \
    git git-lfs \
    build-essential \
    python3-dev \
    sox libsox-dev ffmpeg libsndfile1 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# pipのアップグレード
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorchを段階的にインストール（タイムアウト対策）
RUN pip install --timeout=1000 --retries=5 \
    torch==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --timeout=1000 --retries=5 \
    torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --timeout=1000 --retries=5 \
    torchvision==0.18.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 基本パッケージのインストール
RUN pip install numpy==1.26.4

# cuda-pythonを追加（性能向上のため）
RUN pip install cuda-python>=12.3

# NeMo Toolkitのインストール
RUN pip install nemo_toolkit[asr]

# WORKDIR を設定
WORKDIR /app

# 依存関係ファイルのみをコピー（レイヤーキャッシュ最適化）
COPY requirements.txt ./

# Python依存関係のインストール（requirements.txtが変更されない限りキャッシュされる）
RUN pip install -r requirements.txt

# アプリケーションコード全体のコピー（最後に実行してキャッシュ効率を向上）
COPY . .