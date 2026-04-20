"""
ReachyAI ASR Modal Deployment
==============================
Deploy training and inference on Modal.com GPU cloud.

Usage:
    # Training job
    modal run modal_deploy.py --cv-base-dir /data/cv

    # Serve inference API
    modal deploy modal_deploy.py
"""

import os
import modal

# ====================================================================
# MODAL APP CONFIGURATION
# ====================================================================

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libsndfile1", "libsndfile1-dev", "ffmpeg",
        "sox", "libsox-dev", "git", "wget", "build-essential",
        "cmake", "libboost-all-dev", "zlib1g-dev", "libbz2-dev", "liblzma-dev"
    )
    .pip_install(
        "torch==2.5.1", "torchvision", "torchaudio",
        "transformers==4.52.0", "datasets==3.6.0",
        "accelerate==1.6.0", "peft==0.15.0",
        "evaluate==0.4.3", "jiwer==3.1.0",
        "huggingface-hub==0.30.0",
        "librosa==0.10.2", "soundfile==0.13.1",
        "wandb==0.19.0", "tensorboard==2.18.0",
        "scipy==1.15.0", "pandas==2.2.3", "numpy==1.26.4",
        "fastapi==0.115.0", "uvicorn==0.32.0",
        "pyctcdecode==0.5.0",
    )
    .pip_install("https://github.com/kpu/kenlm/archive/master.zip")
    .pip_install("omnilingual-asr", extra_options="|| true")
    .copy_local_file("config.py", "/app/config.py")
    .copy_local_file("data_pipeline.py", "/app/data_pipeline.py")
    .copy_local_file("models.py", "/app/models.py")
    .copy_local_file("augmentation.py", "/app/augmentation.py")
    .copy_local_file("evaluation.py", "/app/evaluation.py")
    .copy_local_file("lm_fusion.py", "/app/lm_fusion.py")
    .copy_local_file("train.py", "/app/train.py")
    .copy_local_file("inference_server.py", "/app/inference_server.py")
)

# Modal app
app = modal.App("reachy-asr", image=image)

# Persistent volume for model checkpoints
volume = modal.Volume.from_name("reachy-asr-checkpoints", create_if_missing=True)

# ====================================================================
# TRAINING FUNCTION
# ====================================================================

@app.function(
    gpu="A100-40GB",  # or "L40S", "H100", "A10G"
    timeout=86400,    # 24 hours max
    volumes={"/app/output": volume},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb-key")]
)
def train_akan_asr(
    model_family: str = "omnilingual",
    cv_base_dir: str = None,
    num_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 6.25e-6,
    use_lora: bool = True,
    lora_rank: int = 64,
    train_kenlm: bool = True,
    pseudo_label: bool = True,
    resume: str = None,
    seed: int = 42
):
    """
    Train Akan ASR model on Modal GPU.
    
    Args:
        model_family: ASR model to use (omnilingual, mms, xlsr, w2vbert)
        cv_base_dir: Path to Common Voice data in volume
        num_epochs: Training epochs
        batch_size: Per-device batch size
        lr: Learning rate
        use_lora: Use LoRA fine-tuning
        train_kenlm: Train KenLM for decoding
        pseudo_label: Enable pseudo-labeling
        resume: Checkpoint path to resume from
        seed: Random seed
    """
    import subprocess
    
    # Environment
    os.environ["HF_HOME"] = "/app/cache/huggingface"
    os.environ["WANDB_PROJECT"] = "reachy-akan-asr"
    
    # Build command
    cmd = [
        "python", "/app/train.py",
        "--model_family", model_family,
        "--output_dir", "/app/output/reachy-akan-asr",
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
    ]
    
    if cv_base_dir:
        cmd.extend(["--cv_base_dir", cv_base_dir])
    if use_lora:
        cmd.append("--use_lora")
        cmd.extend(["--lora_rank", str(lora_rank)])
    if train_kenlm:
        cmd.append("--train_kenlm")
    if pseudo_label:
        cmd.append("--pseudo_label")
    if resume:
        cmd.extend(["--resume", resume])
    
    # Run training
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode


# ====================================================================
# INFERENCE ASGI APP
# ====================================================================

@app.function(
    gpu="T4",  # Inference can use smaller GPU
    container_idle_timeout=300,  # 5 min idle timeout
    volumes={"/app/output": volume},
    secrets=[modal.Secret.from_name("hf-token")]
)
@modal.asgi_app()
def asr_api():
    """Serve ASR inference API."""
    import uvicorn
    from fastapi import FastAPI
    
    # Import the inference server
    import sys
    sys.path.insert(0, "/app")
    
    from inference_server import app as inference_app
    
    return inference_app


# ====================================================================
# CLI ENTRY POINT
# ====================================================================

@app.local_entrypoint()
def main(
    model_family: str = "omnilingual",
    cv_base_dir: str = None,
    num_epochs: int = 10,
    batch_size: int = 16,
    action: str = "train"  # "train" or "serve"
):
    """Run training or serve inference from CLI."""
    if action == "train":
        result = train_akan_asr.remote(
            model_family=model_family,
            cv_base_dir=cv_base_dir,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        print(f"Training finished with code: {result}")
    
    elif action == "serve":
        print("Deploying inference API...")
        asr_api.deploy()
        print("API deployed at: https://{app_name}--asr-api.modal.run")
