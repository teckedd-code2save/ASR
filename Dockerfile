# =============================================================================
# Dockerfile for Whisper ASR Fine-tuning (RunPod / Generic Cloud GPU)
# =============================================================================
# Model: openai/whisper-medium (or whisper-small)
# Task: Fine-tuning for Twi (Akan) ASR
# Platforms: RunPod, Lambda Labs, Vast.ai, Generic Docker GPU
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="ml-ops-team"
LABEL description="Whisper ASR Fine-tuning Environment"

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies (pinned versions)
# =============================================================================
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    transformers==4.47.1 \
    datasets==3.2.0 \
    accelerate==1.2.1 \
    evaluate==0.4.3 \
    jiwer==3.1.0 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    tensorboard==2.18.0 \
    wandb==0.19.1 \
    librosa==0.10.2.post1 \
    soundfile==0.13.0 \
    matplotlib==3.10.0 \
    gradio==5.12.0 \
    huggingface-hub==0.27.0 \
    hf-transfer==0.1.8 \
    flash-attn==2.7.3 \
    requests==2.32.3 \
    pandas==2.2.3 \
    scipy==1.15.1

# Install faster-whisper for inference optimization
RUN pip install faster-whisper==1.1.1 ctranslate2==4.5.0

# =============================================================================
# Verify GPU Access & Installations
# =============================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
RUN python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
RUN python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# =============================================================================
# Working Directory & Volume Setup
# =============================================================================
RUN mkdir -p /workspace/data /workspace/outputs /workspace/checkpoints /workspace/.cache/huggingface

WORKDIR /workspace

# =============================================================================
# Entry Point
# =============================================================================
# Default: run the training script
# Override with: docker run -v /host/data:/workspace/data <image> python train_script.py
CMD ["python", "/workspace/train_script.py"]
