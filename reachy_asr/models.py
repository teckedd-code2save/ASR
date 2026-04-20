"""
ReachyAI ASR Model Factory
===========================
Multi-backend ASR model support for Akan/Twi:
- Meta Omnilingual ASR (RECOMMENDED - 1,600 languages, Apache 2.0)
- Meta MMS-1B (Akan adapters available)
- XLS-R 300M (CTC-based, good scaling)
- W2v-BERT (contrastive + MLM pretraining)
"""

import os
import warnings
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    # Wav2Vec2 family (CTC-based)
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    AutoModelForCTC, AutoProcessor,
    # Whisper (encoder-decoder, seq2seq)
    WhisperForConditionalGeneration, WhisperProcessor,
    # Generic
    AutoModel, AutoTokenizer, AutoFeatureExtractor
)

from peft import LoraConfig, get_peft_model, TaskType

from config import ModelConfig, TrainingConfig


# ====================================================================
# MODEL LOADING
# ====================================================================

class ASRModelFactory:
    """Factory for loading and configuring ASR models."""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()
    
    # ----------------------------------------------------------------
    # Meta Omnilingual ASR (RECOMMENDED)
    # ----------------------------------------------------------------
    
    def load_omnilingual(
        self,
        model_card: str = "omniASR-CTC-300M",
        device: str = "cuda"
    ) -> Tuple[Any, Any]:
        """
        Load Meta Omnilingual ASR model.
        
        Available models:
        - omniASR-CTC-300M: 325M params, ~2GB VRAM, 96x real-time
        - omniASR-CTC-1B: 975M params, ~3GB VRAM, 48x real-time  
        - omniASR-LLM-300M: 1.6B params, ~5GB VRAM, ~1x real-time (better quality)
        - omniASR-LLM-1B: 2.3B params, ~6GB VRAM
        
        Akan language code: "aka_Latn"
        """
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError:
            print("WARNING: omnilingual-asr not installed. Install with:")
            print("  pip install omnilingual-asr")
            print("Falling back to HuggingFace direct loading...")
            return self._load_omnilingual_hf(model_card, device)
        
        # Use the official pipeline
        pipeline = ASRInferencePipeline(model_card=model_card)
        
        # For training, we need the raw model
        # The pipeline wraps it, so we extract it
        model = pipeline.model
        processor = pipeline.processor
        
        model.to(device)
        return model, processor
    
    def _load_omnilingual_hf(self, model_card: str, device: str):
        """Load Omnilingual model directly from HuggingFace."""
        hf_model_id = f"facebook/{model_card}"
        
        # Try loading as CTC model
        try:
            processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
            model = AutoModelForCTC.from_pretrained(
                hf_model_id,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"CTC loading failed: {e}")
            # Try AutoModel for LLM variants
            processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                hf_model_id,
                trust_remote_code=True
            )
        
        model.to(device)
        return model, processor
    
    # ----------------------------------------------------------------
    # Meta MMS-1B (with Akan adapter)
    # ----------------------------------------------------------------
    
    def load_mms(
        self,
        device: str = "cuda",
        use_adapter: bool = True
    ) -> Tuple[Any, Any]:
        """
        Load Meta MMS-1B model with Akan language adapter.
        
        MMS uses language-specific adapters on top of a shared encoder.
        For Akan, we load the 'aka' adapter.
        
        Reference:
        - https://huggingface.co/facebook/mms-1b-all
        - Akan lang ID: 106 (in MMS vocabulary)
        """
        model_id = self.model_config.mms_model
        lang_id = self.model_config.mms_lang_code  # "aka"
        
        print(f"Loading MMS-1B with {lang_id} adapter...")
        
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        if use_adapter:
            # Load with adapter - only adapter weights are trainable
            model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                target_lang=lang_id,  # Load Akan adapter
                ignore_mismatched_sizes=True
            )
        else:
            # Full model loading
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
        
        model.to(device)
        
        # Print trainable parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
        
        return model, processor
    
    # ----------------------------------------------------------------
    # XLS-R 300M (CTC-based)
    # ----------------------------------------------------------------
    
    def load_xlsr(
        self,
        device: str = "cuda",
        vocab_size: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Load XLS-R 300M model.
        Pre-trained on 436k hours, 128 languages.
        Fine-tuned with CTC loss.
        """
        model_id = self.model_config.xlsr_model
        
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        model_kwargs = {
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "feat_proj_dropout": 0.0,
            "mask_time_prob": 0.075,
            "layerdrop": 0.1,
            "ctc_loss_reduction": "mean",
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        
        if vocab_size:
            model_kwargs["vocab_size"] = vocab_size
        else:
            model_kwargs["vocab_size"] = len(processor.tokenizer)
        
        model = Wav2Vec2ForCTC.from_pretrained(model_id, **model_kwargs)
        model.freeze_feature_encoder()  # Don't fine-tune CNN frontend
        
        model.to(device)
        return model, processor
    
    # ----------------------------------------------------------------
    # W2v-BERT 2.0
    # ----------------------------------------------------------------
    
    def load_w2vbert(
        self,
        device: str = "cuda",
        vocab_size: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Load W2v-BERT 2.0 model.
        Pre-trained on 4.5M hours, 143 languages.
        Combines contrastive predictive coding + masked LM.
        """
        model_id = self.model_config.w2vbert_model
        
        # W2v-BERT uses the same processor as Wav2Vec2
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.075,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=vocab_size or len(processor.tokenizer)
        )
        model.freeze_feature_encoder()
        
        model.to(device)
        return model, processor
    
    # ----------------------------------------------------------------
    # Whisper (with fallback for languages not in vocabulary)
    # ----------------------------------------------------------------
    
    def load_whisper(
        self,
        model_size: str = "medium",
        language: Optional[str] = None,  # None = no language token
        device: str = "cuda"
    ) -> Tuple[Any, Any]:
        """
        Load Whisper model.
        NOTE: Whisper does NOT have Akan in its vocabulary.
        We use it WITHOUT language token for multilingual fine-tuning.
        """
        model_id = f"openai/whisper-{model_size}"
        
        # Load without language specification
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        # Override language token if specified
        if language is None:
            # No language token - multilingual mode
            model.generation_config.language = None
            model.generation_config.task = "transcribe"
            model.generation_config.forced_decoder_ids = None
            model.config.forced_decoder_ids = None
            print("  Whisper loaded WITHOUT language token (multilingual mode)")
        
        model.to(device)
        return model, processor
    
    # ----------------------------------------------------------------
    # Main Factory Method
    # ----------------------------------------------------------------
    
    def load_model(
        self,
        model_family: Optional[str] = None,
        device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        """
        Load model based on configured family.
        
        Returns:
            (model, processor, model_family)
        """
        family = model_family or self.model_config.model_family
        
        print(f"\nLoading ASR model: {family}")
        print("=" * 50)
        
        if family == "omnilingual":
            model, processor = self.load_omnilingual(
                model_card=self.model_config.omnilingual_ctc_model,
                device=device
            )
        elif family == "omnilingual_llm":
            model, processor = self.load_omnilingual(
                model_card=self.model_config.omnilingual_llm_model,
                device=device
            )
        elif family == "mms":
            model, processor = self.load_mms(
                device=device,
                use_adapter=self.model_config.use_adapter
            )
        elif family == "xlsr":
            model, processor = self.load_xlsr(device=device)
        elif family == "w2vbert":
            model, processor = self.load_w2vbert(device=device)
        elif family == "whisper":
            model, processor = self.load_whisper(
                language=None,  # No language token for Akan
                device=device
            )
        else:
            raise ValueError(f"Unknown model family: {family}")
        
        print(f"  Model family: {family}")
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print("=" * 50)
        
        return model, processor, family


# ====================================================================
# LoRA CONFIGURATION
# ====================================================================

def apply_lora(
    model: Any,
    config: ModelConfig,
    task_type: TaskType = TaskType.CTC
) -> Any:
    """
    Apply LoRA (Low-Rank Adaptation) to the model for efficient fine-tuning.
    
    Benefits:
    - Reduces trainable parameters by ~90-95%
    - Enables training on smaller GPUs
    - Often generalizes better (less overfitting)
    """
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=task_type,
        modules_to_save=["lm_head", "classifier"]  # Also train head layers
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


# ====================================================================
# PROGRESSIVE TRAINING
# ====================================================================

def setup_progressive_training(
    model: Any,
    phase: str = "decoder_only"
) -> Any:
    """
    Setup model for progressive training.
    
    Phase 1: "decoder_only" - Freeze encoder, train only decoder/CTC head
    Phase 2: "full" - Unfreeze all layers for full fine-tuning
    """
    if phase == "decoder_only":
        # Freeze encoder
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            # Whisper-style
            for param in model.model.encoder.parameters():
                param.requires_grad = False
            for param in model.model.decoder.parameters():
                param.requires_grad = True
        elif hasattr(model, "wav2vec2"):
            # Wav2Vec2-style: freeze feature extractor + encoder
            for param in model.wav2vec2.parameters():
                param.requires_grad = False
            # Only train CTC head / LM head
            for param in model.lm_head.parameters():
                param.requires_grad = True
        
        print("Progressive training: Phase 1 (decoder/head only)")
        
    elif phase == "full":
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        
        # But keep feature encoder frozen (XLS-R best practice)
        if hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "feature_extractor"):
            for param in model.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        print("Progressive training: Phase 2 (full fine-tune, frozen CNN)")
    
    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")
    
    return model


# ====================================================================
# TEST
# ====================================================================

if __name__ == "__main__":
    factory = ASRModelFactory()
    
    # Test loading each model type
    for family in ["mms", "xlsr", "w2vbert"]:
        print(f"\n{'='*60}")
        print(f"Testing: {family}")
        print(f"{'='*60}")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor, fam = factory.load_model(family, device=device)
            print(f"SUCCESS: {family}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: {e}")
