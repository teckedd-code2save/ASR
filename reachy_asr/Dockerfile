# ====================================================================
# ReachyAI ASR Training Container
# ================================
# Multi-stage build for optimized training on GPU cloud platforms.
# Supports: RunPod, Modal, Vast.ai, Lambda Labs
#
# Build:
#   docker build -t reachy-asr .
#
# Run (training):
#   docker run --gpus all -v $(pwd)/data:/data -e HF_TOKEN=$HF_TOKEN \
#     reachy-asr python train.py --model_family omnilingual
#
# Run (inference):
#   docker run --gpus all -p 8000:8000 reachy-asr python inference_server.py
# ====================================================================

# ----------------------------------------------------------------
# Stage 1: Base with system dependencies
# ----------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    wget \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-dev \
    build-essential \
    cmake \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------
# Stage 2: Python dependencies
# ----------------------------------------------------------------
FROM base AS deps

WORKDIR /app

# Install PyTorch (CUDA 12.1)
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install \
    transformers==4.52.0 \
    datasets==3.6.0 \
    accelerate==1.6.0 \
    peft==0.15.0 \
    evaluate==0.4.3 \
    jiwer==3.1.0 \
    huggingface-hub==0.30.0

# Audio processing
RUN pip install \
    librosa==0.10.2 \
    soundfile==0.13.1 \
    audiomentations==0.40.0

# Training & experiment tracking
RUN pip install \
    wandb==0.19.0 \
    tensorboard==2.18.0 \
    scipy==1.15.0 \
    pandas==2.2.3 \
    numpy==1.26.4

# Optional: KenLM + pyctcdecode for LM fusion
RUN pip install \
    pyctcdecode==0.5.0 \
    https://github.com/kpu/kenlm/archive/master.zip

# Optional: Omnilingual ASR
RUN pip install \
    omnilingual-asr \
    || echo "omnilingual-asr not available, skipping"

# ----------------------------------------------------------------
# Stage 3: Application code
# ----------------------------------------------------------------
FROM deps AS app

WORKDIR /app

# Copy source code
COPY config.py .
COPY data_pipeline.py .
COPY models.py .
COPY augmentation.py .
COPY evaluation.py .
COPY lm_fusion.py .
COPY train.py .
COPY inference_server.py .

# Create directories
RUN mkdir -p /app/cache /app/preprocessed /app/output /app/data

# Environment variables
ENV HF_HOME=/app/cache/huggingface
ENV HF_DATASETS_CACHE=/app/cache/datasets
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV WANDB_PROJECT=reachy-akan-asr

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || exit 1

# Default command
CMD ["python", "train.py", "--help"]
