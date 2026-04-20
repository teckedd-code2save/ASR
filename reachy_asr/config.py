"""
ReachyAI ASR Training Configuration
====================================
Centralized config for Akan/Twi ASR training pipeline.
Supports multiple model backends: Omnilingual, MMS, XLS-R, W2v-BERT.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Primary datasets
    waxal_dataset: str = "google/WaxalNLP"
    waxal_config: str = "aka_asr"  # Akan ASR subset
    farmerline_dataset: str = "ghananlpcommunity/twi_dataset_2.0_farmerline"
    
    # Common Voice - downloaded from Mozilla Data Collective
    cv_version: str = "cv-corpus-24.0-2025-12-05"
    cv_language: str = "tw"  # Twi
    cv_tsv_train: str = "train.tsv"
    cv_tsv_test: str = "test.tsv"
    cv_clips_dir: str = "clips"
    
    # Sampling
    sampling_rate: int = 16000
    max_audio_duration_sec: float = 30.0
    min_audio_duration_sec: float = 1.0
    
    # Splits
    val_ratio: float = 0.1  # 10% of train for validation
    test_ratio: float = 0.0  # Use provided test sets
    
    # Caching
    cache_dir: str = "./cache"
    preprocessed_dir: str = "./preprocessed"
    
    # Pseudo-labeling
    waxal_unlabeled_split: str = "aka_unlabeled"  # 109k unlabeled clips
    pseudo_label_confidence_threshold: float = 0.85
    pseudo_label_max_clips: int = 50_000  # Process max 50k at a time


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Model family selection
    model_family: str = "omnilingual"  # "omnilingual", "mms", "xlsr", "w2vbert"
    
    # Specific model checkpoints
    omnilingual_ctc_model: str = "facebook/omniASR-CTC-300M"
    omnilingual_llm_model: str = "facebook/omniASR-LLM-300M"
    mms_model: str = "facebook/mms-1b-all"  # Has Akan adapters!
    xlsr_model: str = "facebook/wav2vec2-xls-r-300m"
    w2vbert_model: str = "facebook/w2v-bert-2.0"
    
    # Language codes
    # Omnilingual uses ISO 639-3 + script: "aka_Latn" for Akan (Latin script)
    # MMS uses ISO 639-3: "aka" for Akan
    # XLS-R/W2v-BERT use language ID from preprocessor
    omnilingual_lang_code: str = "aka_Latn"  # Akan in Latin script
    mms_lang_code: str = "aka"  # MMS Akan language ID
    
    # Training strategy
    use_adapter: bool = True  # For MMS: use adapter-based fine-tuning
    lora_rank: int = 64  # For LoRA fine-tuning
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"
    ])
    
    # Progressive training
    progressive_training: bool = True
    freeze_encoder_steps: int = 1500  # Freeze encoder for first N steps
    decoder_only_lr: float = 1e-4  # Higher LR for decoder-only phase


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for Akan ASR."""
    # Batch config
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2  # Effective batch = 32
    
    # Optimization
    learning_rate: float = 6.25e-6  # Conservative for fine-tuning
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    lr_scheduler_type: str = "cosine"  # Cosine decay beats linear
    warmup_ratio: float = 0.1  # 10% of total steps
    num_train_epochs: float = 10  # Full dataset coverage
    max_steps: int = -1  # Use epochs instead (-1 = auto)
    
    # Precision
    fp16: bool = False
    bf16: bool = True  # bf16 preferred on modern GPUs
    tf32: bool = True
    
    # Memory
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True  # Reduces padding waste
    
    # Checkpointing
    output_dir: str = "./reachy-akan-asr"
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 250
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 25
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    run_name: Optional[str] = None
    
    # Generation (for eval)
    predict_with_generate: bool = True
    generation_max_length: int = 225
    generation_num_beams: int = 5  # Beam search for eval
    
    # Reproducibility
    seed: int = 42
    data_seed: int = 42
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "checkpoint"
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # SpecAugment
    use_spec_augment: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100
    n_freq_masks: int = 2
    n_time_masks: int = 2
    
    # Speed perturbation
    use_speed_perturbation: bool = True
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.1])
    speed_perturb_prob: float = 0.5
    
    # Noise injection
    use_noise_injection: bool = False  # Enable if MUSAN available
    noise_snr_db_range: tuple = (5, 20)
    noise_prob: float = 0.3
    
    # RIR (Room Impulse Response)
    use_rir: bool = False  # Enable if RIR data available
    rir_prob: float = 0.3


@dataclass
class LMConfig:
    """Language model configuration for decoding fusion."""
    # KenLM
    train_kenlm: bool = True
    kenlm_order: int = 4  # 4-gram
    kenlm_binary: str = "./kenlm/build/bin/lmplz"
    kenlm_build_binary: str = "./kenlm/build/bin/build_binary"
    
    # Text corpus for LM
    lm_text_source: str = "transcriptions"  # Use all training transcriptions
    lm_additional_text: Optional[str] = None  # Path to additional Twi text
    
    # Decoding parameters (tuned on validation set)
    lm_alpha: float = 0.6  # LM weight
    lm_beta: float = 0.4   # Word insertion bonus
    
    # pyctcdecode
    use_pyctcdecode: bool = True
    beam_width: int = 100


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Normalization
    use_whisper_normalizer: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True
    normalize_unicode: bool = True
    
    # Metrics
    compute_wer: bool = True
    compute_cer: bool = True
    
    # Per-domain evaluation
    per_domain_eval: bool = True  # Eval separately on Waxal, Farmerline, CV
    
    # Confidence intervals
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Error analysis
    error_analysis: bool = True
    analyze_diacritics: bool = True  # Track ɛ, ɔ, ŋ errors


# ==================== Environment Setup ====================

def setup_environment():
    """Set environment variables for optimal training."""
    # HuggingFace cache
    os.environ.setdefault("HF_HOME", "./cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "./cache/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "./cache/transformers")
    
    # Tokenizers parallelism
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # PyTorch optimizations
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # W&B
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "reachy-akan-asr"
    
    # Verify required env vars
    required = []
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"WARNING: Missing env vars: {missing}")


# Default configs
default_data_config = DataConfig()
default_model_config = ModelConfig()
default_training_config = TrainingConfig()
default_aug_config = AugmentationConfig()
default_lm_config = LMConfig()
default_eval_config = EvalConfig()
