#!/usr/bin/env python3
"""
Modal.com Deployment Script for Whisper ASR Fine-tuning
Usage:
    modal run modal_train.py
    modal deploy modal_train.py  # For persistent deployment
"""

import modal
import os

# =============================================================================
# Modal Configuration
# =============================================================================

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "libgomp1", "git", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.47.1",
        "datasets==3.2.0",
        "accelerate==1.2.1",
        "evaluate==0.4.3",
        "jiwer==3.1.0",
        "peft==0.14.0",
        "bitsandbytes==0.45.0",
        "tensorboard==2.18.0",
        "wandb==0.19.1",
        "librosa==0.10.2.post1",
        "soundfile==0.13.0",
        "matplotlib==3.10.0",
        "requests==2.32.3",
        "pandas==2.2.3",
        "scipy==1.15.1",
        "numpy==1.26.4",
        "huggingface-hub==0.27.0",
        "tqdm==4.67.1",
        "python-dotenv==1.0.1",
    )
    .pip_install("flash-attn==2.7.3", "faster-whisper==1.1.1")
    .env({
        "HF_HOME": "/vol/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/vol/.cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# Create a persistent volume for datasets and checkpoints
dataset_volume = modal.Volume.from_name("whisper-datasets", create_if_missing=True)
output_volume = modal.Volume.from_name("whisper-outputs", create_if_missing=True)

# Modal App definition
app = modal.App("whisper-twi-asr", image=image)

# Secrets for API tokens (create these in Modal dashboard)
hf_secret = modal.Secret.from_name("huggingface-token", required=False)
wandb_secret = modal.Secret.from_name("wandb-api-key", required=False)

# =============================================================================
# Training Function
# =============================================================================

@app.function(
    gpu="A100-40GB",  # Options: A100-40GB, A100-80GB, L40S, H100
    cpu=8,
    memory=32768,  # 32GB RAM
    timeout=3600,  # 1 hour timeout
    volumes={
        "/vol/datasets": dataset_volume,
        "/vol/outputs": output_volume,
    },
    secrets=[hf_secret, wandb_secret],
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    ),
)
def train_whisper(
    model_id: str = "openai/whisper-medium",
    output_dir: str = "/vol/outputs/whisper-twi-asr",
    checkpoint_dir: str = "/vol/outputs/checkpoints",
    max_steps: int = 1000,
    batch_size: int = 16,
    gradient_accumulation: int = 2,
    learning_rate: float = 1e-5,
    use_lora: bool = False,
    lora_rank: int = 32,
):
    """
    Train Whisper model for Twi (Akan) ASR.

    Args:
        model_id: HuggingFace model identifier
        output_dir: Where to save the final model
        checkpoint_dir: Where to save intermediate checkpoints
        max_steps: Maximum training steps
        batch_size: Per-device training batch size
        gradient_accumulation: Gradient accumulation steps
        learning_rate: Peak learning rate
        use_lora: Whether to use LoRA instead of full fine-tuning
        lora_rank: LoRA rank (if use_lora=True)
    """
    import torch
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import load_dataset, DatasetDict, Audio, interleave_datasets, Dataset
    import evaluate
    import wandb
    import logging
    import sys
    from datetime import datetime
    import requests
    import pandas as pd
    import re
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import os

    # =================================================================
    # Logging Configuration
    # =================================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"/vol/outputs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # =================================================================
    # Environment Validation
    # =================================================================
    logger.info("=" * 60)
    logger.info("Starting Whisper Twi ASR Training")
    logger.info(f"Model: {model_id} | Steps: {max_steps} | Batch: {batch_size}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    logger.info(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
    logger.info("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Set it in Modal secrets.")

    from huggingface_hub import login
    login(token=hf_token)
    logger.info("Authenticated with HuggingFace Hub")

    # =================================================================
    # Initialize W&B
    # =================================================================
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.init(
            project="whisper-twi-asr",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_id": model_id,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
            },
        )
        logger.info("Weights & Biases initialized")

    # =================================================================
    # Load Model and Processor
    # =================================================================
    logger.info(f"Loading model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language="Twi", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Twi", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Apply LoRA if requested
    if use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        logger.info(f"Applying LoRA (rank={lora_rank})")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.generation_config.language = "Twi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # Required for gradient checkpointing
    logger.info("Model loaded successfully")

    # =================================================================
    # Load Datasets
    # =================================================================
    logger.info("Loading datasets...")
    SAMPLING_RATE = 16000
    COLS_TO_REMOVE = ["id", "speaker_id", "language", "gender"]

    # WaxalNLP - load to disk first for reliability
    waxal_train = load_dataset("google/WaxalNLP", "aka_asr", split="train")
    waxal_test = load_dataset("google/WaxalNLP", "aka_asr", split="test")

    for split, ds in [("train", waxal_train), ("test", waxal_test)]:
        cols = [c for c in COLS_TO_REMOVE if c in ds.column_names]
        if cols:
            waxal_train = waxal_train.remove_columns(cols) if split == "train" else waxal_train
            waxal_test = waxal_test.remove_columns(cols) if split == "test" else waxal_test

    waxal_train = waxal_train.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    waxal_test = waxal_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Farmerline
    farmerline = load_dataset("ghananlpcommunity/twi_dataset_2.0_farmerline")

    # Interleave WaxalNLP + Farmerline
    from datasets import interleave_datasets
    train_ds = interleave_datasets([waxal_train, farmerline["train"]], seed=42)
    test_ds = interleave_datasets([waxal_test, farmerline["test"]], seed=42)

    logger.info(f"Train dataset size: {len(train_ds)} samples")
    logger.info(f"Test dataset size: {len(test_ds)} samples")

    # =================================================================
    # Preprocessing
    # =================================================================
    CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\"\%\'\"\\]'

    def clean_transcription(text: str) -> str:
        return re.sub(CHARS_TO_IGNORE_REGEX, '', text).lower().strip() + " "

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=SAMPLING_RATE
        ).input_features[0]
        batch["labels"] = tokenizer(
            clean_transcription(batch["transcription"])
        ).input_ids
        return batch

    logger.info("Preprocessing datasets...")
    cols_to_remove = ["audio", "transcription"]
    train_ds = train_ds.map(prepare_dataset, remove_columns=cols_to_remove)
    test_ds = test_ds.map(prepare_dataset, remove_columns=cols_to_remove)

    # =================================================================
    # Data Collator
    # =================================================================
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # =================================================================
    # Metrics
    # =================================================================
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # =================================================================
    # Training Arguments (optimized for cloud)
    # =================================================================
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        gradient_accumulation_steps=gradient_accumulation,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        tf32=True if torch.cuda.is_available() else False,
        learning_rate=learning_rate,
        warmup_steps=100,
        max_steps=max_steps,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=5,
        logging_steps=25,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=225,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["wandb"] if wandb_api_key else ["tensorboard"],
        run_name=f"whisper-twi-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        push_to_hub=True,
        hub_strategy="checkpoint",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # =================================================================
    # Trainer Setup with Checkpoint Resumption
    # =================================================================
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Check for existing checkpoints to resume from
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # =================================================================
    # Save & Push to Hub
    # =================================================================
    logger.info("Training complete! Saving model...")
    trainer.save_model(output_dir)

    kwargs = {
        "dataset_tags": ["google/WaxalNLP", "ghananlpcommunity/twi_dataset_2.0_farmerline"],
        "dataset": "WaxalNLP aka_asr, Twi Dataset 2.0 Farmerline",
        "language": "tw",
        "model_name": "Whisper Medium Twi ASR",
        "finetuned_from": model_id,
        "tasks": "automatic-speech-recognition",
    }
    trainer.push_to_hub(**kwargs)
    processor.push_to_hub(output_dir.split("/")[-1])
    logger.info("Model pushed to HuggingFace Hub!")

    if wandb_api_key:
        wandb.finish()

    return {"status": "success", "output_dir": output_dir, "model": model_id}


# =============================================================================
# Inference Endpoint (separate from training)
# =============================================================================

@app.function(
    gpu="A10G",  # Smaller GPU for inference
    image=image,
    secrets=[hf_secret],
    timeout=120,
)
def transcribe(audio_path: str, model_id: str = "teckedd/whisper-medium-twi-asr") -> str:
    """Inference endpoint for transcription."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_path, language="tw", task="transcribe")
    return " ".join([s.text for s in segments])


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    with app.run():
        result = train_whisper.remote()
        print(f"Training result: {result}")
