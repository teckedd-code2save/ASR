#!/usr/bin/env python3
"""
ReachyAI ASR Training Pipeline
================================
Optimized training for Akan/Twi ASR with multi-model support.

Usage:
    # Quick start with defaults (Omnilingual CTC-300M)
    python train.py

    # MMS with adapter fine-tuning
    python train.py --model_family mms --use_adapter

    # XLS-R with LoRA
    python train.py --model_family xlsr --use_lora --lora_rank 32

    # With pseudo-labeling
    python train.py --pseudo_label --pseudo_threshold 0.85

    # Resume from checkpoint
    python train.py --resume ./reachy-akan-asr/checkpoint-5000

    # Full config
    python train.py \
        --model_family omnilingual \
        --cv_base_dir /path/to/cv-corpus-24.0-2025-12-05/tw \
        --output_dir ./reachy-akan-asr \
        --num_epochs 15 \
        --batch_size 16 \
        --lr 6.25e-6 \
        --use_lora \
        --train_kenlm \
        --wandb_project reachy-akan-asr
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import (
    Trainer, TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

# Local imports
from config import (
    DataConfig, ModelConfig, TrainingConfig,
    AugmentationConfig, LMConfig, EvalConfig,
    setup_environment
)
from data_pipeline import DataPipeline, normalize_text
from models import ASRModelFactory, apply_lora, setup_progressive_training
from augmentation import AugmentedDataCollatorCTC
from evaluation import create_compute_metrics_fn, ASREvaluator
from lm_fusion import setup_lm_fusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ====================================================================
# ARGUMENT PARSING
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ReachyAI Akan/Twi ASR Training Pipeline"
    )
    bool_action = argparse.BooleanOptionalAction
    
    # Model selection
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_family", type=str,
        default="omnilingual",
        choices=["omnilingual", "omnilingual_llm", "mms", "xlsr", "w2vbert", "whisper"],
        help="ASR model family to use")
    model_group.add_argument("--use_lora", action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning")
    model_group.add_argument("--lora_rank", type=int, default=64)
    model_group.add_argument("--use_adapter", action="store_true",
        help="Use MMS language adapter (for MMS only)")
    model_group.add_argument("--progressive", action=bool_action, default=True,
        help="Use progressive training (freeze encoder first)")
    
    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--cv_base_dir", type=str, default=None,
        help="Path to Common Voice Twi corpus (downloaded from Mozilla Data Collective)")
    data_group.add_argument("--cache_dir", type=str, default="./cache")
    data_group.add_argument("--val_ratio", type=float, default=0.1)
    
    # Pseudo-labeling
    pseudo_group = parser.add_argument_group("Pseudo-Labeling")
    pseudo_group.add_argument("--pseudo_label", action="store_true",
        help="Generate pseudo-labels for unlabeled WaxalNLP data")
    pseudo_group.add_argument("--pseudo_threshold", type=float, default=0.85)
    pseudo_group.add_argument("--pseudo_max_clips", type=int, default=50000)
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--output_dir", type=str, default="./reachy-akan-asr")
    train_group.add_argument("--num_epochs", type=int, default=10)
    train_group.add_argument("--batch_size", type=int, default=16)
    train_group.add_argument("--grad_accum", type=int, default=2)
    train_group.add_argument("--lr", type=float, default=6.25e-6)
    train_group.add_argument("--weight_decay", type=float, default=0.01)
    train_group.add_argument("--warmup_ratio", type=float, default=0.1)
    train_group.add_argument("--max_grad_norm", type=float, default=1.0)
    train_group.add_argument("--scheduler", type=str, default="cosine",
        choices=["linear", "cosine", "polynomial", "constant"])
    train_group.add_argument("--seed", type=int, default=42)
    
    # Augmentation
    aug_group = parser.add_argument_group("Augmentation")
    aug_group.add_argument("--spec_augment", action=bool_action, default=True)
    aug_group.add_argument("--speed_perturb", action=bool_action, default=False)
    aug_group.add_argument("--noise_inject", action=bool_action, default=False)
    
    # LM Fusion
    lm_group = parser.add_argument_group("Language Model")
    lm_group.add_argument("--train_kenlm", action="store_true",
        help="Train KenLM n-gram for decoding fusion")
    lm_group.add_argument("--kenlm_order", type=int, default=4)
    
    # Evaluation
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval_per_domain", action=bool_action, default=True)
    
    # Infrastructure
    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--resume", type=str, default=None,
        help="Resume from checkpoint path")
    infra_group.add_argument("--bf16", action=bool_action, default=True)
    infra_group.add_argument("--fp16", action=bool_action, default=False)
    infra_group.add_argument("--wandb_project", type=str, default="reachy-akan-asr")
    infra_group.add_argument("--push_to_hub", action="store_true", default=False)
    infra_group.add_argument("--hub_model_id", type=str, default=None)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to use for training")
    
    return parser.parse_args()


# ====================================================================
# MAIN TRAINING PIPELINE
# ====================================================================

def main():
    args = parse_args()
    
    # Setup
    setup_environment()
    set_seed(args.seed)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info("=" * 70)
    logger.info("REACHYAI AKAN/TWI ASR TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_family}")
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 70)
    
    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    
    data_config = DataConfig(
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        pseudo_label_confidence_threshold=args.pseudo_threshold,
        pseudo_label_max_clips=args.pseudo_max_clips
    )
    
    model_config = ModelConfig(
        model_family=args.model_family,
        use_adapter=args.use_adapter,
        lora_rank=args.lora_rank,
        progressive_training=args.progressive
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.fp16 and device == "cuda",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        run_name=f"{args.model_family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    aug_config = AugmentationConfig(
        use_spec_augment=args.spec_augment,
        use_speed_perturbation=args.speed_perturb,
        use_noise_injection=args.noise_inject
    )
    
    lm_config = LMConfig(
        train_kenlm=args.train_kenlm,
        kenlm_order=args.kenlm_order
    )
    
    eval_config = EvalConfig(
        per_domain_eval=args.eval_per_domain
    )
    
    # ----------------------------------------------------------------
    # Load Model & Processor
    # ----------------------------------------------------------------
    
    factory = ASRModelFactory(model_config)
    model, processor, model_family = factory.load_model(
        model_family=args.model_family,
        device=device
    )
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Applying LoRA...")
        model = apply_lora(model, model_config)
    
    # Setup progressive training
    if args.progressive:
        model = setup_progressive_training(model, phase="decoder_only")
    
    # ----------------------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------------------
    
    pipeline = DataPipeline(data_config)
    
    dataset = pipeline.build_dataset(
        processor=processor,
        model_family=model_family,
        cv_base_dir=args.cv_base_dir,
        skip_preprocessing=False
    )
    
    # Pseudo-labeling (Phase 2 data augmentation)
    if args.pseudo_label:
        logger.info("\n" + "=" * 70)
        logger.info("PSEUDO-LABELING")
        logger.info("=" * 70)
        
        try:
            # Load unlabeled WaxalNLP data
            from datasets import load_dataset
            unlabeled = load_dataset(
                data_config.waxal_dataset,
                data_config.waxal_unlabeled_split,
                split="train",
                trust_remote_code=True
            )
            
            if len(unlabeled) > 0:
                logger.info(f"Unlabeled clips: {len(unlabeled)}")
                
                # Use a copy of model for pseudo-labeling
                from data_pipeline import generate_pseudo_labels
                pseudo_ds = generate_pseudo_labels(
                    model=model,
                    processor=processor,
                    unlabeled_dataset=unlabeled,
                    config=data_config,
                    device=device
                )
                
                if len(pseudo_ds) > 0:
                    # Preprocess pseudo-labeled data
                    pseudo_processed = pipeline.preprocess_dataset(
                        pseudo_ds, processor, "pseudo", model_family
                    )
                    
                    # Combine with training data
                    from datasets import concatenate_datasets
                    original_train = dataset["train"]
                    dataset["train"] = concatenate_datasets([
                        original_train, pseudo_processed
                    ]).shuffle(seed=args.seed)
                    
                    logger.info(f"Combined train: {len(dataset['train'])} samples")
        except Exception as e:
            logger.warning(f"Pseudo-labeling failed: {e}")
            logger.warning("Continuing without pseudo-labels")
    
    # ----------------------------------------------------------------
    # Setup Data Collator
    # ----------------------------------------------------------------
    
    is_ctc_model = model_family in ["omnilingual", "mms", "xlsr", "w2vbert"]
    
    if is_ctc_model:
        data_collator = AugmentedDataCollatorCTC(
            processor=processor,
            padding=True,
            augmentation_config=aug_config,
            apply_augmentation=True
        )
    else:
        # Whisper uses default data collator
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            processor=processor,
            model=model,
            padding=True
        )
    
    # ----------------------------------------------------------------
    # Setup Compute Metrics
    # ----------------------------------------------------------------
    
    compute_metrics = create_compute_metrics_fn(processor, eval_config)
    
    # ----------------------------------------------------------------
    # Setup Training Arguments
    # ----------------------------------------------------------------
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hf_training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Training
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        
        # Precision
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        tf32=training_config.tf32,
        
        # Memory
        gradient_checkpointing=training_config.gradient_checkpointing,
        dataloader_num_workers=training_config.dataloader_num_workers,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        group_by_length=training_config.group_by_length,
        
        # Checkpointing
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        
        # Evaluation
        evaluation_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        
        # Logging
        logging_strategy=training_config.logging_strategy,
        logging_steps=training_config.logging_steps,
        report_to=training_config.report_to,
        run_name=training_config.run_name,
        
        # Reproducibility
        seed=training_config.seed,
        data_seed=training_config.data_seed,
        
        # Hub
        push_to_hub=training_config.push_to_hub,
        hub_model_id=training_config.hub_model_id,
    )
    
    # ----------------------------------------------------------------
    # Setup Trainer
    # ----------------------------------------------------------------
    
    callbacks = []
    
    # Early stopping
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    ))
    
    trainer = Trainer(
        model=model,
        args=hf_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else None,
    )
    
    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")
    logger.info(f"Test samples: {len(dataset['test'])}")
    logger.info(f"Epochs: {training_config.num_train_epochs}")
    logger.info(f"Batch size (per device): {training_config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Scheduler: {training_config.lr_scheduler_type}")
    logger.info(f"Max grad norm: {training_config.max_grad_norm}")
    logger.info(f"BF16: {training_config.bf16}")
    logger.info("=" * 70)
    
    # Phase 1: Decoder-only training
    if args.progressive:
        logger.info("\n--- Phase 1: Decoder/Head Training ---")
        trainer.train(resume_from_checkpoint=args.resume)
        
        # Phase 2: Full fine-tuning
        logger.info("\n--- Phase 2: Full Model Fine-Tuning ---")
        model = setup_progressive_training(model, phase="full")
        
        # Update trainer with unfrozen model
        trainer.model = model
        
        # Train more epochs
        hf_training_args.num_train_epochs = max(3, args.num_epochs // 3)
        hf_training_args.learning_rate = args.lr / 2  # Lower LR for full fine-tune
        trainer.args = hf_training_args
        
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=args.resume)
    
    # ----------------------------------------------------------------
    # Final Evaluation
    # ----------------------------------------------------------------
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    
    # Evaluate on test set
    test_results = trainer.evaluate(
        eval_dataset=dataset["test"],
        metric_key_prefix="test"
    )
    
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # Per-domain evaluation
    if args.eval_per_domain and "source" in dataset["test"].column_names:
        evaluator = ASREvaluator(eval_config)
        
        # Get predictions on test set
        predictions_output = trainer.predict(dataset["test"])
        pred_ids = predictions_output.predictions
        if hasattr(pred_ids, "shape") and len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(predictions_output.label_ids == -100,
                           processor.tokenizer.pad_token_id,
                           predictions_output.label_ids)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        sources = dataset["test"]["source"]
        
        per_domain = evaluator.compute_per_domain_metrics(pred_str, label_str, sources)
        
        logger.info("\nPer-Domain Results:")
        for domain, metrics in per_domain.items():
            logger.info(f"  {domain}: {metrics}")
    
    # ----------------------------------------------------------------
    # Save Final Model
    # ----------------------------------------------------------------
    
    logger.info("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    
    # Save LoRA adapter if used
    if args.use_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir / "final" / "lora_adapter"))
    
    logger.info(f"Model saved to: {output_dir / 'final'}")
    
    # ----------------------------------------------------------------
    # KenLM Fusion (post-training)
    # ----------------------------------------------------------------
    
    if args.train_kenlm and is_ctc_model:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING KENLM FOR DECODING FUSION")
        logger.info("=" * 70)
        
        # Get all training transcriptions
        train_texts = [normalize_text(t) for t in dataset["train"]["transcription"]
                      if "transcription" in dataset["train"].column_names]
        
        if train_texts:
            lm_decoder = setup_lm_fusion(
                processor=processor,
                train_transcriptions=train_texts,
                output_dir=str(output_dir / "lm"),
                lm_config=lm_config
            )
            
            if lm_decoder:
                logger.info("KenLM fusion ready!")
        else:
            logger.warning("No training transcriptions found for LM training")
    
    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model: {output_dir / 'final'}")
    logger.info(f"Logs: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
