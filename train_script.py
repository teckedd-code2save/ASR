#!/usr/bin/env python3
"""
Production-Ready Training Script for Whisper ASR Fine-tuning
Supports: RunPod, Modal, Vast.ai, Lambda Labs, local GPU
Features:
  - Environment validation
  - Checkpoint resumption
  - Persistent storage support
  - W&B experiment tracking
  - LoRA support
  - Logging
  - Fault tolerance
"""

import os
import sys
import re
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import torch
import numpy as np
import requests
import pandas as pd
from datasets import (
    load_dataset,
    DatasetDict,
    Audio,
    interleave_datasets,
    Dataset,
)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import login
import evaluate

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper ASR Fine-tuning")

    # Model
    parser.add_argument("--model_id", type=str, default="openai/whisper-medium",
                        help="HuggingFace model identifier")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank")

    # Training
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Paths
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/outputs/whisper-twi-asr",
                        help="Output directory for model")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/workspace/checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--data_dir", type=str, default="/workspace/data",
                        help="Directory for cached datasets")
    parser.add_argument("--cache_dir", type=str,
                        default="/workspace/.cache/huggingface",
                        help="HuggingFace cache directory")

    # Logging & Tracking
    parser.add_argument("--report_to", type=str, default="tensorboard,wandb",
                        help="Reporting integrations (comma-separated)")
    parser.add_argument("--wandb_project", type=str, default="whisper-twi-asr",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for tracking")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if available")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")

    return parser.parse_args()

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)

# =============================================================================
# Environment Validation
# =============================================================================

def validate_environment(args, logger):
    """Validate all required environment variables and system config."""
    errors = []
    warnings_list = []

    logger.info("=" * 60)
    logger.info("Environment Validation")
    logger.info("=" * 60)

    # Check Python version
    import platform
    py_version = platform.python_version()
    logger.info(f"Python version: {py_version}")
    if int(py_version.split(".")[1]) < 10:
        errors.append(f"Python 3.10+ required, got {py_version}")

    # Check CUDA
    if not torch.cuda.is_available():
        errors.append("CUDA not available. GPU is required for training.")
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")

        # VRAM sanity check for whisper-medium with batch=16
        min_vram_map = {
            "openai/whisper-medium": 14,
            "openai/whisper-small": 8,
            "openai/whisper-large": 24,
        }
        for model_key, min_vram in min_vram_map.items():
            if model_key in args.model_id:
                if vram_gb < min_vram:
                    errors.append(
                        f"{args.model_id} needs ~{min_vram}GB VRAM, "
                        f"got {vram_gb:.1f}GB. Use a smaller model, LoRA, or smaller batch."
                    )
                break

    # Check tokens
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        errors.append("HF_TOKEN environment variable not set")
    else:
        logger.info("HF_TOKEN: [SET]")

    wandb_key = os.environ.get("WANDB_API_KEY")
    report_targets = [r.strip() for r in args.report_to.split(",")]
    if "wandb" in report_targets and not wandb_key:
        warnings_list.append("WANDB_API_KEY not set — disabling W&B tracking")
        report_targets.remove("wandb")
        args.report_to = ",".join(report_targets)

    mozilla_key = os.environ.get("MOZILLA_APIKEY")
    if not mozilla_key:
        warnings_list.append("MOZILLA_APIKEY not set — Common Voice data will be skipped")

    # Check directories
    for d in [args.output_dir, args.checkpoint_dir, args.data_dir, args.cache_dir]:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory ready: {d}")

    # Check write permissions
    for d in [args.output_dir, args.checkpoint_dir]:
        test_file = os.path.join(d, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            errors.append(f"Cannot write to {d}: {e}")

    # Check transformers version
    import transformers
    logger.info(f"Transformers: {transformers.__version__}")
    logger.info(f"Datasets: {__import__('datasets').__version__}")

    # Report results
    if warnings_list:
        for w in warnings_list:
            logger.warning(f"[WARN] {w}")

    if errors:
        logger.error("VALIDATION FAILED:")
        for e in errors:
            logger.error(f"  [ERROR] {e}")
        raise RuntimeError(f"Environment validation failed with {len(errors)} error(s)")

    logger.info("Environment validation PASSED")
    logger.info("=" * 60)

# =============================================================================
# Data Loading with Caching
# =============================================================================

def load_and_cache_datasets(args, logger):
    """Load datasets with persistent caching to disk."""
    SAMPLING_RATE = 16000
    COLS_TO_REMOVE = ["id", "speaker_id", "language", "gender"]
    asr_data = DatasetDict()

    # WaxalNLP
    logger.info("Loading WaxalNLP dataset...")
    waxal_train = load_dataset(
        "google/WaxalNLP", "aka_asr", split="train",
        cache_dir=args.cache_dir, trust_remote_code=True,
    )
    waxal_test = load_dataset(
        "google/WaxalNLP", "aka_asr", split="test",
        cache_dir=args.cache_dir, trust_remote_code=True,
    )
    logger.info(f"  WaxalNLP: {len(waxal_train)} train, {len(waxal_test)} test")

    for split_name, ds in [("train", waxal_train), ("test", waxal_test)]:
        cols = [c for c in COLS_TO_REMOVE if c in ds.column_names]
        if cols:
            ds = ds.remove_columns(cols)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        asr_data[split_name] = ds

    # Farmerline
    logger.info("Loading Farmerline dataset...")
    farmerline = load_dataset(
        "ghananlpcommunity/twi_dataset_2.0_farmerline",
        cache_dir=args.cache_dir, trust_remote_code=True,
    )
    logger.info(f"  Farmerline: {len(farmerline['train'])} train, {len(farmerline['test'])} test")

    # Interleave
    logger.info("Interleaving datasets...")
    train_ds = interleave_datasets(
        [asr_data["train"], farmerline["train"]],
        seed=args.seed,
    )
    test_ds = interleave_datasets(
        [asr_data["test"], farmerline["test"]],
        seed=args.seed,
    )

    # Common Voice (if API key available)
    mozilla_key = os.environ.get("MOZILLA_APIKEY")
    if mozilla_key:
        logger.info("Attempting Common Voice download...")
        cv_train, cv_test = download_common_voice(mozilla_key, args, logger)
        if cv_train is not None:
            train_ds = interleave_datasets([train_ds, cv_train], seed=args.seed)
            test_ds = interleave_datasets([test_ds, cv_test], seed=args.seed)
            logger.info("Common Voice merged successfully")
    else:
        logger.info("Skipping Common Voice (no MOZILLA_APIKEY)")

    logger.info(f"Final: {len(train_ds)} train samples, {len(test_ds)} test samples")
    return train_ds, test_ds


def download_common_voice(api_key, args, logger, max_retries=3):
    """Download Common Voice with retry logic."""
    import tarfile
    import time

    url = "https://datacollective.mozillafoundation.org/api/datasets/cmj8u3py800stnxxbljgesvle/download"

    for attempt in range(max_retries):
        try:
            logger.info(f"  API request attempt {attempt + 1}/{max_retries}...")
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            download_url = response.json().get("downloadUrl")

            if not download_url:
                raise ValueError("No downloadUrl in response")

            tarball = os.path.join(args.data_dir, "mcv-scripted-tw-v24.0.tar.gz")
            if not os.path.exists(tarball):
                logger.info(f"  Downloading from {download_url[:50]}...")
                r = requests.get(download_url, stream=True, timeout=300)
                r.raise_for_status()
                with open(tarball, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"  Downloaded {os.path.getsize(tarball) / 1e6:.1f} MB")

            extract_dir = os.path.join(args.data_dir, "cv-corpus-24.0")
            if not os.path.exists(extract_dir):
                logger.info("  Extracting tarball...")
                with tarfile.open(tarball, "r:gz") as tar:
                    tar.extractall(args.data_dir)

            base_dir = os.path.join(args.data_dir, "cv-corpus-24.0-2025-12-05", "tw")
            clips_dir = os.path.join(base_dir, "clips")

            train_df = pd.read_csv(os.path.join(base_dir, "train.tsv"), sep="\t")
            test_df = pd.read_csv(os.path.join(base_dir, "test.tsv"), sep="\t")

            train_df["audio"] = train_df["path"].apply(lambda p: os.path.join(clips_dir, p))
            test_df["audio"] = test_df["path"].apply(lambda p: os.path.join(clips_dir, p))

            train_df = train_df[["audio", "sentence"]].rename(columns={"sentence": "transcription"})
            test_df = test_df[["audio", "sentence"]].rename(columns={"sentence": "transcription"})

            cv_train = Dataset.from_pandas(train_df).cast_column(
                "audio", Audio(sampling_rate=16000)
            )
            cv_test = Dataset.from_pandas(test_df).cast_column(
                "audio", Audio(sampling_rate=16000)
            )

            return cv_train, cv_test

        except Exception as e:
            logger.error(f"  Common Voice download failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.warning("  Common Voice data will be skipped")
                return None, None

# =============================================================================
# Data Collator
# =============================================================================

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

# =============================================================================
# Main Training Function
# =============================================================================

def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    os.environ["HF_HOME"] = args.cache_dir

    # Run name
    if args.run_name is None:
        args.run_name = f"whisper-twi-{args.model_id.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Validate
    validate_environment(args, logger)

    # Login to HF
    login(token=os.environ["HF_TOKEN"])

    # Initialize W&B
    report_targets = [r.strip() for r in args.report_to.split(",") if r.strip()]
    if "wandb" in report_targets:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "model_id": args.model_id,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank,
            },
        )

    # =================================================================
    # Load Model
    # =================================================================
    logger.info(f"Loading model: {args.model_id}")

    processor = WhisperProcessor.from_pretrained(args.model_id, language="Twi", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language="Twi", task="transcribe")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {dtype}")

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        logger.info(f"Applying LoRA (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
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
    model.config.use_cache = False

    # =================================================================
    # Load Data
    # =================================================================
    train_ds, test_ds = load_and_cache_datasets(args, logger)

    # Preprocess
    CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\"\%\'\"\\]'

    def clean_transcription(text):
        return re.sub(CHARS_TO_IGNORE_REGEX, '', text).lower().strip() + " "

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = tokenizer(clean_transcription(batch["transcription"])).input_ids
        return batch

    logger.info("Preprocessing datasets...")
    cols_to_remove = [c for c in train_ds.column_names if c not in ["input_features", "labels"]]
    train_ds = train_ds.map(prepare_dataset, remove_columns=cols_to_remove)
    test_ds = test_ds.map(prepare_dataset, remove_columns=cols_to_remove)

    # =================================================================
    # Collator & Metrics
    # =================================================================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

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
    # Training Arguments
    # =================================================================
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size // 2, 1),
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        tf32=True if torch.cuda.is_available() else False,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=25,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=225,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=report_targets,
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        push_to_hub=args.push_to_hub,
        hub_strategy="checkpoint",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # =================================================================
    # Checkpoint Resumption
    # =================================================================
    last_checkpoint = None
    if args.resume:
        checkpoint_dirs = [
            args.checkpoint_dir,
            args.output_dir,
        ]
        for chk_dir in checkpoint_dirs:
            if os.path.isdir(chk_dir):
                checkpoints = [
                    os.path.join(chk_dir, d)
                    for d in os.listdir(chk_dir)
                    if d.startswith("checkpoint-")
                ]
                if checkpoints:
                    last_checkpoint = sorted(
                        checkpoints,
                        key=lambda x: int(x.split("-")[-1])
                    )[-1]
                    break

    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        logger.info("Starting training from scratch")

    # =================================================================
    # Train
    # =================================================================
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # =================================================================
    # Save & Push
    # =================================================================
    logger.info("Saving final model...")
    trainer.save_model(args.output_dir)

    if args.push_to_hub:
        logger.info("Pushing to HuggingFace Hub...")
        kwargs = {
            "dataset_tags": ["google/WaxalNLP", "ghananlpcommunity/twi_dataset_2.0_farmerline"],
            "dataset": "WaxalNLP aka_asr, Twi Dataset 2.0 Farmerline",
            "language": "tw",
            "model_name": f"Whisper {args.model_id.split('/')[-1]} Twi ASR",
            "finetuned_from": args.model_id,
            "tasks": "automatic-speech-recognition",
        }
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(args.output_dir.split("/")[-1])
        logger.info("Model pushed to Hub successfully!")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
