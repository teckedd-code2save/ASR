# Strategic Roadmap: Achieving WER < 10% on Twi (Akan) ASR

## Executive Summary

This roadmap provides a comprehensive, phased strategy to achieve **WER < 10%** on Twi (Akan) ASR, starting from the current notebook setup. The current setup has critical flaws that, when corrected, combined with advanced techniques, can dramatically improve performance.

**Current State Assessment:**
- Model: Whisper-medium (769M params) - reasonable choice
- Data: ~33 hours combined (WaxalNLP 14.3h + Farmerline 3.8h + Common Voice Twi ~15h)
- Training: Only 1000 steps with lr=1e-5
- **Critical Bug**: Language token set to "yoruba" instead of the closest valid option
- **No validation split**: Test set used as evaluation data (data leakage risk)
- **No data augmentation**: Missing SpecAugment, speed perturbation, noise injection
- **No text normalization**: WER computed without proper normalization
- **Akan/Twi is NOT in Whisper's language list** (confirmed via source code inspection)

**Benchmark Context:**
| Dataset | Best Reported WER | Model | Notes |
|---------|------------------|-------|-------|
| Multilingual African Speech (Akan) | 46.74% | Whisper large-v3 | Zero-shot baseline |
| Ashesi Financial Inclusion (Twi) | 6.25% | Sahara V2* | Proprietary model |
| FLEURS (Twi) | 12.46% | Sahara V2* | Best open benchmark |
| FLEURS (Twi) | 39.83% | Gemini-3.0 flash | Commercial LLM |
| FLEURS (Twi) | 50.55% | MMS-1b all | Open source |
| African LRL Benchmark (Akan, 100h) | ~31% | MMS | Fine-tuned plateau |
| African LRL Benchmark (Akan, 100h) | ~30% | W2v-BERT | Fine-tuned plateau |
| Swahili ASR (similar Bantu LRL) | 16.88% | Whisper | 400h fine-tuning |
| Swahili ASR (Zindi challenge) | 17.81% | Whisper Turbo + LoRA | Winning solution |

> **Critical Insight**: Akan/Twi does not appear in Whisper's LANGUAGES dictionary. The notebook's use of `"yoruba"` is a reasonable but suboptimal fallback. The best strategy is to use the closest related language or let Whisper auto-detect.

---

## Phase 1: Quick Wins (Expected WER reduction: 40-60% relative)
**Timeline: 1-2 days | Implementation effort: 4-8 hours**

### 1.1 Fix Language Token Configuration
| Aspect | Details |
|--------|---------|
| **Issue** | Notebook sets `language="yoruba"` - Yoruba is a different Niger-Congo language |
| **Root Cause** | Akan/Twi is NOT in Whisper's 99-language support list (confirmed via source code grep) |
| **Fix Options** | (a) Use `"akan"` if transformers library maps it (try first); (b) Use no language token (auto-detect); (c) Keep `"yoruba"` as closest related language; (d) Use `"english"` for code-mixed scenarios |
| **Expected WER Impact** | **-15% to -25% relative WER reduction** (using proper language routing) |
| **Implementation** | Test each option with 100 samples; select best based on WER |

**Code Fix:**
```python
# Try these options and benchmark each:
# Option 1: No language specification (auto-detect)
model.generation_config.language = None
# Option 2: Use yoruba (current - closest related Niger-Congo language)
model.generation_config.language = "yoruba"
# Option 3: Use swahili (another African language in Whisper's training data)
model.generation_config.language = "swahili"
```

### 1.2 Implement Proper Text Normalization for WER
| Aspect | Details |
|--------|---------|
| **Issue** | Current cleaning only strips punctuation via regex; no WER-specific normalization |
| **Impact** | Inflated WER due to case differences, extra whitespace, diacritic mismatches |
| **Expected WER Impact** | **-10% to -20% relative WER reduction** (normalization artifact removal) |

**Code Fix:**
```python
import jiwer

# Proper normalization for Twi WER computation
def normalize_twi_text(text):
    """Normalize Twi text for fair WER computation."""
    transforms = jiwer.Compose([
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
        jiwer.transforms.Lowercase(),
        # Optional: normalize Unicode diacritics (e.g., combining characters)
    ])
    return transforms(text)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Apply normalization BEFORE WER computation
    pred_str = [normalize_twi_text(s) for s in pred_str]
    label_str = [normalize_twi_text(s) for s in label_str]
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

### 1.3 Fix Train/Validation/Test Splits
| Aspect | Details |
|--------|---------|
| **Issue** | Test set used as eval_dataset (no validation split); streaming datasets make splitting harder |
| **Impact** | Cannot do early stopping properly; risk of overfitting to test set; no unbiased final evaluation |
| **Expected Impact** | Enables proper model selection and early stopping |

**Code Fix:**
```python
from datasets import DatasetDict

# For non-streaming datasets: use proper 80/10/10 split
# For streaming: materialize a validation split
# Create validation split from train (hold out 10%)
# Reserve test set ONLY for final evaluation

asr_data = DatasetDict({
    "train": combined_train,      # 80% of labeled data
    "validation": combined_val,   # 10% of labeled data  
    "test": combined_test,        # 10% of labeled data (NEVER used during training)
})

# Use validation for training eval, test ONLY for final report
trainer = Seq2SeqTrainer(
    eval_dataset=asr_data["validation"],  # NOT test!
    ...
)
```

### 1.4 Increase Training Steps Dramatically
| Aspect | Details |
|--------|---------|
| **Issue** | Only 1000 steps with effective batch size 32 = 32k samples seen; ~33h of audio |
| **Best Practice** | For low-resource ASR, train for **many more steps** (often 5k-20k+) |
| **Research Evidence** | Swahili Whisper tiny: WER dropped from 83% to 31% when training increased from 5 to 100 epochs |
| **Expected WER Impact** | **-20% to -35% relative WER reduction** |

**Recommended Hyperparameters:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-akan-asr",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch = 32
    gradient_checkpointing=True,
    fp16=True,
    # === LEARNING RATE (key change) ===
    learning_rate=1e-5,  # Keep current (good for Whisper fine-tuning)
    # === INCREASED TRAINING ===
    warmup_steps=500,    # Increased from 100
    max_steps=5000,      # INCREASED from 1000 to 5000
    # === SCHEDULER ===
    lr_scheduler_type="cosine",  # Better than linear for fine-tuning
    # === EVALUATION ===
    eval_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=3,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    # === REGULARIZATION ===
    weight_decay=0.01,  # ADDED - prevents overfitting
    # === GENERATION ===
    predict_with_generate=True,
    generation_max_length=225,
    report_to=["tensorboard"],
)
```

### 1.5 Use bf16 Instead of fp16 (if available)
| Aspect | Details |
|--------|---------|
| **Issue** | fp16 can cause gradient underflow; bf16 has same range as fp32 |
| **Expected Impact** | More stable training, especially with low-resource data |

---

## Phase 2: Data Strategy & Scaling (Expected WER reduction: 30-50% relative)
**Timeline: 1-2 weeks | Implementation effort: 20-40 hours**

### 2.1 Data Size Analysis: Is ~33 Hours Enough?

| Dataset | Hours | Clips | Avg Duration | Domain |
|---------|-------|-------|-------------|--------|
| WaxalNLP aka_asr | ~14.3h | 5,147 | ~10s | General conversational |
| Farmerline 2.0 | ~3.8h | 3,432 | ~4s | Agricultural/financial |
| Common Voice Twi v24 | ~15h (est.) | ~13,500 | ~4s | Read speech (crowdsourced) |
| **TOTAL** | **~33h** | **~22,000** | **~5.4s avg** | Mixed |

**Verdict: 33 hours is on the LOW end for WER < 10%. Research evidence:**
- W2v-BERT on Akan: WER plateaus at ~30% even with 100h of data
- Whisper on Swahili: WER ~17% with 400h of fine-tuning data
- XLS-R on Akan: WER improves from 55.5% to 30.7% with just 20h + LM
- **Rule of thumb: 50-100+ hours typically needed for sub-10% WER on low-resource languages**

**Action: Target at least 50-80 hours through augmentation and data collection.**

### 2.2 SpecAugment (Highest Priority Augmentation)
| Aspect | Details |
|--------|---------|
| **Technique** | Frequency masking + Time masking on mel spectrograms |
| **Evidence** | SpecAugment is standard for Whisper training; frequency masking gives biggest gains |
| **Expected WER Impact** | **-15% to -25% relative WER reduction** |
| **Implementation** | Via transformers `DataCollatorSpeechSeq2SeqWithPadding` or custom augmentation |

**Code Implementation:**
```python
from transformers import Seq2SeqTrainer
import torch
import numpy as np

class SpecAugmentDataCollator(DataCollatorSpeechSeq2SeqWithPadding):
    """Data collator with SpecAugment applied to input features."""
    
    def __init__(self, *args, freq_mask_param=27, time_mask_param=100, 
                 n_freq_masks=2, n_time_masks=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        # Apply SpecAugment to input_features
        input_features = batch["input_features"]  # (B, 80, 3000)
        B, F, T = input_features.shape
        
        for b in range(B):
            # Frequency masking
            for _ in range(self.n_freq_masks):
                f = np.random.randint(0, self.freq_mask_param + 1)
                f0 = np.random.randint(0, F - f + 1)
                input_features[b, f0:f0+f, :] = 0
            
            # Time masking (with adaptive upper bound p=0.2)
            for _ in range(self.n_time_masks):
                max_t = min(self.time_mask_param, int(0.2 * T))
                t = np.random.randint(0, max_t + 1)
                t0 = np.random.randint(0, T - t + 1)
                input_features[b, :, t0:t0+t] = 0
        
        batch["input_features"] = input_features
        return batch

# Use "LibriSpeech Double" policy for stronger augmentation on low-resource data:
# freq_mask_param=27, time_mask_param=100, n_freq_masks=2, n_time_masks=2
```

### 2.3 Speed Perturbation (Time-Stretching)
| Aspect | Details |
|--------|---------|
| **Technique** | Resample audio at 0.9x and 1.1x speed, creating 3x data |
| **Evidence** | Speed perturbation is the most common augmentation for ASR; ~5-10% WER improvement |
| **Expected WER Impact** | **-10% to -15% relative WER reduction** |
| **Data Multiplication** | 3x (original + 0.9x + 1.1x) |

**Code Implementation:**
```python
import librosa
import numpy as np

def apply_speed_perturbation(audio_array, sr=16000):
    """Apply speed perturbation (0.9x, 1.0x, 1.1x)."""
    speeds = [0.9, 1.1]
    augmented = []
    for speed in speeds:
        stretched = librosa.effects.time_stretch(audio_array, rate=speed)
        augmented.append({"array": stretched, "sampling_rate": sr})
    return augmented

# Apply during dataset preparation
def prepare_dataset_with_augmentation(batch):
    audio = batch["audio"]
    samples = [{"array": audio["array"], "sampling_rate": audio["sampling_rate"]}]
    
    # 50% chance of speed perturbation
    if np.random.random() < 0.5:
        samples.extend(apply_speed_perturbation(audio["array"]))
    
    # Process each variant
    batch["input_features"] = [
        feature_extractor(s["array"], sampling_rate=s["sampling_rate"]).input_features[0]
        for s in samples
    ]
    batch["labels"] = [tokenizer(clean_transcription(batch["transcription"])).input_ids] * len(samples)
    return batch
```

### 2.4 Noise Injection (MUSAN / Background Noise)
| Aspect | Details |
|--------|---------|
| **Technique** | Mix clean audio with background noise at various SNRs (5dB, 10dB, 15dB) |
| **Evidence** | Critical for robustness; Common Voice data is clean read speech, real world is noisy |
| **Expected WER Impact** | **-5% to -10% relative WER reduction** (especially on noisy test data) |
| **Source** | MUSAN corpus (free) or synthetic noise |

### 2.5 Room Impulse Response (RIR) Simulation
| Aspect | Details |
|--------|---------|
| **Technique** | Convolve audio with room impulse responses to simulate reverberation |
| **Expected WER Impact** | **-3% to -8% relative WER reduction** (on reverberant data) |

### 2.6 Leverage Unlabeled WaxalNLP Data (Self-Supervised / Pseudo-Labeling)
| Aspect | Details |
|--------|---------|
| **Opportunity** | WaxalNLP has ~109k unlabeled Akan clips (~300+ hours unlabeled!) |
| **Approach** | (1) Train initial model on labeled data; (2) Generate pseudo-labels for unlabeled; (3) Filter high-confidence; (4) Retrain on combined |
| **Expected WER Impact** | **-15% to -30% relative WER reduction** (pseudo-labeling can add massive effective training data) |
| **Caution** | Filter pseudo-labels by confidence score; noisy pseudo-labels can hurt performance |

**Pseudo-Labeling Pipeline:**
```python
# Step 1: Train initial model on labeled data (~33h)
# Step 2: Generate pseudo-labels for 109k unlabeled clips
def generate_pseudo_labels(model, unlabeled_dataset, confidence_threshold=0.8):
    pseudo_labeled = []
    for batch in unlabeled_dataset:
        input_features = batch["input_features"]
        with torch.no_grad():
            outputs = model.generate(
                input_features,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # Compute average token confidence
            scores = torch.stack(outputs.scores, dim=1).softmax(-1)
            token_confidences = scores.max(-1).values.mean(-1)
            
            if token_confidences.item() > confidence_threshold:
                text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
                pseudo_labeled.append({"audio": batch["audio"], "transcription": text})
    return pseudo_labeled

# Step 3: Combine labeled + high-confidence pseudo-labeled
# Step 4: Retrain from scratch or continue training
```

### 2.7 TTS Data Augmentation (Synthetic Data)
| Aspect | Details |
|--------|---------|
| **Technique** | Use TTS to generate synthetic Twi speech from text corpora |
| **Source** | Coqui TTS, yourTTS, or pretrained multilingual TTS with Twi support |
| **Expected WER Impact** | **-5% to -15% relative WER reduction** (if high-quality TTS available) |
| **Caveat** | Synthetic data can introduce acoustic mismatch; use as supplement only |

### 2.8 Combined Data Augmentation Summary

| Technique | Data Multiplication | Expected WER Gain | Effort (hours) |
|-----------|-------------------|-------------------|----------------|
| SpecAugment | 1x (online) | -15% to -25% | 2 |
| Speed perturbation (0.9x, 1.1x) | 3x | -10% to -15% | 3 |
| Noise injection (MUSAN) | 2-3x | -5% to -10% | 4 |
| RIR simulation | 2x | -3% to -8% | 4 |
| Pseudo-labeling (109k unlabeled) | +300h effective | -15% to -30% | 8 |
| TTS augmentation | +10-50h | -5% to -15% | 6 |
| **TOTAL EFFECTIVE DATA** | **~400-500h** | **Combined: -40% to -60%** | **~27h** |

---

## Phase 3: Model Strategy & Architecture
**Timeline: 3-7 days | Implementation effort: 15-30 hours**

### 3.1 Model Size Selection

| Model | Params | Pros | Cons | Recommendation |
|-------|--------|------|------|----------------|
| Whisper-small | 244M | Faster training, less overfitting risk | Lower capacity | Baseline; may not reach <10% WER |
| **Whisper-medium** | **769M** | **Good capacity/data tradeoff** | **Slower, more memory** | **CURRENT - KEEP THIS** |
| Whisper-large-v2 | 1550M | Best zero-shot performance | Risk of overfitting on 33h | Consider if data > 50h |
| Whisper-large-v3 | 1550M | Best multilingual performance | Same overfitting risk | Best if data > 80h |
| Whisper-turbo | 809M | Large-v3 distilled, 8x faster | Not for fine-tuning | For inference only |

**Recommendation: Stick with Whisper-medium for 33-50h of data. Upgrade to large-v2/v3 only when effective data exceeds 80h.**

### 3.2 Fine-Tuning Strategy: Progressive Training
| Aspect | Details |
|--------|---------|
| **Technique** | Freeze encoder → fine-tune decoder → unfreeze encoder → full fine-tune |
| **Rationale** | Decoder adapts language token distribution first; encoder learns acoustic features later |
| **Expected WER Impact** | **-5% to -10% relative WER reduction** vs. full fine-tuning from start |
| **Evidence** | Standard practice for low-resource fine-tuning; prevents catastrophic forgetting |

**Implementation:**
```python
# Phase A: Decoder-only fine-tuning (first 1500 steps)
for param in model.model.encoder.parameters():
    param.requires_grad = False
for param in model.model.decoder.parameters():
    param.requires_grad = True

trainer.train(resume_from_checkpoint=None)  # Train decoder only

# Phase B: Full fine-tuning (remaining 3500 steps)
for param in model.model.encoder.parameters():
    param.requires_grad = True

# Continue training with lower learning rate
training_args.learning_rate = 5e-6  # Halve LR for full fine-tuning
trainer.train(resume_from_checkpoint="./checkpoint-1500")
```

### 3.3 LoRA vs Full Fine-Tuning

| Method | Trainable Params | Pros | Cons | When to Use |
|--------|-----------------|------|------|-------------|
| **Full fine-tune** | 100% (769M) | Best performance ceiling | Overfitting risk, more compute | When data > 50h |
| **LoRA (r=32)** | ~5% (~38M) | Less overfitting, faster, less memory | Slight performance gap | When data < 50h or limited compute |
| **LoRA (r=64)** | ~10% (~77M) | Good middle ground | - | **Recommended for 33h data** |

**Recommendation for current data size (~33h): Use LoRA with rank=64, alpha=128, targeting all linear layers in attention (Q, K, V, O) and feed-forward.**

```python
from peft import LoraConfig, get_peft_model, PeftModel

lora_config = LoraConfig(
    r=64,                    # Rank (higher for very low-resource)
    lora_alpha=128,          # Scaling factor = 2x rank
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", 
                     "fc1", "fc2"],  # All linear layers
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["embed_tokens", "lm_head"],  # Also train embeddings
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~5-10% trainable
```

> **Research Evidence**: LoRA achieves comparable WER to full fine-tuning for low-resource languages (Frisian: 23.5% vs 22.4% W2v-BERT; LoRA within 1-3% of full fine-tune). LoRA actually GENERALIZES better due to less overfitting.

### 3.4 Distil-Whisper for Faster Inference
| Aspect | Details |
|--------|---------|
| **Use Case** | After training, distill to smaller model for deployment |
| **Evidence** | Distil-Whisper performs within 1% WER of large model |
| **Note** | Distil-Whisper English-only for now; multilingual distillation requires custom training |

---

## Phase 4: Advanced Training Techniques
**Timeline: 2-5 days | Implementation effort: 10-20 hours**

### 4.1 Multi-Task Learning (ASR + Translation)
| Aspect | Details |
|--------|---------|
| **Technique** | Train model to transcribe Twi AND translate Twi to English simultaneously |
| **Rationale** | Translation task provides additional learning signal; English transcriptions easier to obtain |
| **Expected WER Impact** | **-5% to -10% relative WER reduction** for ASR task |
| **Implementation** | Alternate between transcribe and translate tasks per batch |

### 4.2 Adversarial Training / Domain Adaptation
| Aspect | Details |
|--------|---------|
| **Technique** | Add domain classifier on encoder outputs; adversarially train to remove domain-specific features |
| **Rationale** | WaxalNLP, Farmerline, and Common Voice have different acoustic characteristics |
| **Expected Impact** | Better generalization across datasets |

---

## Phase 5: Decoding & Inference Optimization
**Timeline: 2-3 days | Implementation effort: 8-15 hours**

### 5.1 Beam Search Optimization
| Aspect | Details |
|--------|---------|
| **Current** | Likely greedy decoding (num_beams=1) |
| **Optimal** | Beam search with 3-5 beams |
| **Expected WER Impact** | **-5% to -10% relative WER reduction** |
| **Tradeoff** | Slower inference |

```python
generation_config = {
    "num_beams": 5,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,  # Prevent repetition hallucinations
    "length_penalty": 1.0,
}
```

### 5.2 KenLM N-gram Language Model Fusion
| Aspect | Details |
|--------|---------|
| **Technique** | Train n-gram LM on Twi text corpus; use shallow fusion during beam search decoding |
| **Evidence** | LM fusion gives 5-15% relative WER improvement, especially effective in mid-resource (10-50h) conditions |
| **Expected WER Impact** | **-10% to -20% relative WER reduction** |
| **Steps** | (1) Collect all Twi text from training transcriptions; (2) Train 4-gram or 5-gram KenLM; (3) Tune LM weight (alpha) and word insertion bonus (beta) on validation set |

```python
# Train KenLM on Twi text corpus
# 1. Extract all transcriptions to a text file
# 2. Build KenLM:
#    lmplz -o 4 < twi_text.txt > twi_4gram.arpa
#    build_binary twi_4gram.arpa twi_4gram.bin

# Use with pyctcdecode or flashlight for LM fusion
from pyctcdecode import build_ctcdecoder

# For Whisper (seq2seq), use shallow fusion:
# The LM scores are added to model logits during beam search
# alpha: LM weight (typically 0.5-2.0)
# beta: word insertion bonus (typically 0.0-2.0)
```

### 5.3 Temperature Scheduling
| Aspect | Details |
|--------|---------|
| **Technique** | Use temperature < 1.0 during decoding for more focused predictions |
| **Expected WER Impact** | **-2% to -5% relative WER reduction** |

```python
# Lower temperature = more confident predictions
generation_config = {
    "temperature": 0.7,  # Below 1.0 for more focused decoding
    "do_sample": False,  # Keep deterministic with beam search
}
```

### 5.4 Ensemble Decoding
| Aspect | Details |
|--------|---------|
| **Technique** | Train multiple models with different seeds/augmentations; ensemble at decoding |
| **Expected WER Impact** | **-5% to -10% relative WER reduction** |
| **Cost** | 3-5x training compute |

---

## Phase 6: Error Analysis Framework
**Timeline: Ongoing | Implementation effort: 8-12 hours initial setup**

### 6.1 Per-Dataset Error Analysis
```python
# Evaluate separately on each dataset's test split
datasets = ["waxalnlp", "farmerline", "common_voice"]
for dataset_name in datasets:
    test_data = load_test_split(dataset_name)
    wer = evaluate_model(model, test_data)
    print(f"{dataset_name}: WER = {wer:.2f}%")
```

### 6.2 Error Categorization
```python
import jiwer

def analyze_errors(predictions, references):
    """Categorize errors into insertions, deletions, substitutions."""
    output = jiwer.process_characters(references, predictions)
    
    # Get per-word breakdown
    details = jiwer.compute_measures(references, predictions)
    
    print(f"WER: {details['wer']*100:.2f}%")
    print(f"Substitutions: {details['substitutions']}")
    print(f"Deletions: {details['deletions']}")
    print(f"Insertions: {details['insertions']}")
    print(f"Hits: {details['hits']}")
    
    return details
```

### 6.3 Confusion Matrices for Common Twi Phonemes
- Focus on: tone markers (high, low, falling), vowel harmony, nasalized vowels
- Track: diacritic preservation rate (e, ɛ, o, ɔ are critical in Twi)

### 6.4 Diacritic and Tone Error Analysis
```python
def analyze_diacritic_errors(pred, ref):
    """Analyze errors in Twi-specific characters."""
    twi_special_chars = set('ɛɔɛ̃ɔ̃ŋƆƐŊ')
    pred_chars = set(pred)
    ref_chars = set(ref)
    
    missing = ref_chars - pred_chars
    extra = pred_chars - ref_chars
    
    diacritic_missing = missing & twi_special_chars
    diacritic_extra = extra & twi_special_chars
    
    return {
        "missing_diacritics": diacritic_missing,
        "extra_diacritics": diacritic_extra,
    }
```

---

## Summary: Expected WER Trajectory

Starting from current baseline (estimated **WER ~50-80%** based on benchmarks and the bugs identified):

| Phase | Action | Relative WER Reduction | Estimated WER |
|-------|--------|----------------------|---------------|
| **Baseline** | Current setup (with bugs) | - | ~60-80% |
| **P1.1** | Fix language token | -15% to -25% | ~45-68% |
| **P1.2** | Text normalization | -10% to -20% | ~36-61% |
| **P1.4** | Increase training steps (1k→5k) | -20% to -35% | ~23-49% |
| **P1 Combined** | All Phase 1 fixes | -40% to -60% | **~24-48%** |
| **P2.2** | SpecAugment | -15% to -25% | ~18-41% |
| **P2.3** | Speed perturbation | -10% to -15% | ~15-37% |
| **P2.6** | Pseudo-labeling (109k unlabeled) | -15% to -30% | **~10-26%** |
| **P2 Combined** | All Phase 2 augmentation | -40% to -60% | **~10-19%** |
| **P3.2** | Progressive training | -5% to -10% | ~9-17% |
| **P3.3** | LoRA fine-tuning | -5% to -10% | ~8-15% |
| **P5.1** | Beam search (5 beams) | -5% to -10% | ~7-14% |
| **P5.2** | KenLM 4-gram fusion | -10% to -20% | **~6-11%** |
| **FINAL** | All phases combined | -85% to -93% | **~6-10%** |

> **Realistic Target: WER 8-12% within 2-3 weeks, WER < 10% achievable with full pseudo-labeling and LM fusion.**

---

## Realistic Timeline to WER < 10%

| Week | Focus | Deliverable | Expected WER |
|------|-------|-------------|--------------|
| **Week 1** | Phase 1 (all quick fixes) | Fixed notebook, proper evaluation | 25-45% |
| **Week 1-2** | Phase 2 (data augmentation) | Augmented pipeline, SpecAugment, speed perturb | 15-25% |
| **Week 2-3** | Phase 2 (pseudo-labeling) | 109k unlabeled data labeled, retrained model | 10-18% |
| **Week 3** | Phase 3 (model optimization) | LoRA/Progressive training, beam search | 8-15% |
| **Week 3-4** | Phase 5 (LM fusion) | KenLM trained, decoding optimized | **6-10%** |

---

## Compute Cost Estimates

| Phase | GPU Hours (A100) | Approximate Cost ($3/hr) |
|-------|-----------------|-------------------------|
| Phase 1 (quick fixes) | 8-16 hrs | $25-50 |
| Phase 2 (augmentation pipeline) | 4-8 hrs (dev) | $12-25 |
| Phase 2 (training with augmentation, 5000 steps) | 20-40 hrs | $60-120 |
| Phase 2 (pseudo-labeling 109k clips) | 10-20 hrs | $30-60 |
| Phase 3 (LoRA fine-tuning) | 15-30 hrs | $45-90 |
| Phase 5 (KenLM + beam search tuning) | 4-8 hrs | $12-25 |
| **TOTAL** | **60-120 GPU-hours** | **$180-360** |

---

## Key Risk Factors

1. **Akan not in Whisper's training data**: If Whisper has very little Akan pre-training data, even extensive fine-tuning may plateau early. Mitigation: Use pseudo-labeling to maximize effective data.
2. **Domain mismatch across datasets**: WaxalNLP (conversational), Farmerline (agricultural), Common Voice (read speech) have very different acoustic properties. Mitigation: Domain-adversarial training or per-domain models.
3. **Lack of Twi text corpus for LM**: KenLM needs large Twi text. If unavailable, use all training transcripts + any Twi text found online. Mitigation: Train character-level LM if word-level is insufficient.
4. **Twi tonal system**: Twi is a tonal language; Whisper may not capture tone well. Mitigation: Error analysis focused on tone-marked characters; consider custom acoustic features.

---

## Immediate Action Items (Next 48 Hours)

1. [ ] **Fix language token**: Test "akan", None (auto-detect), "yoruba", "swahili" - benchmark 100 samples each
2. [ ] **Add text normalization**: Implement jiwer-based normalization in compute_metrics
3. [ ] **Create validation split**: Hold out 10% of train for validation; reserve test for final eval only
4. [ ] **Increase training to 5000 steps** with cosine scheduler and weight decay
5. [ ] **Add SpecAugment**: Implement frequency/time masking in data collator
6. [ ] **Add speed perturbation**: 0.9x and 1.1x augmentation
7. [ ] **Implement proper error logging**: Track per-dataset WER, error categories
8. [ ] **Start pseudo-labeling pipeline**: Process WaxalNLP 109k unlabeled clips

---

## References

1. [^23^] Benchmarking ASR Models for African Languages (2025) - XLS-R, Whisper, W2v-BERT, MMS comparison
2. [^24^] Fine-tuning Whisper on Low-Resource Languages for Real-World Applications (2024)
3. [^25^] Full Fine-Tuning vs. Parameter-Efficient Adaptation for African LRLs (2026)
4. [^26^] Fine-tuning Whisper Tiny for Swahili ASR (2025)
5. [^28^] Probing Multilingual and Accent Robustness of Speech LLMs (EACL 2026)
6. [^45^] Low-Resource Speech Recognition by Fine-Tuning Whisper with Optuna-LoRA (2025)
7. [^46^] Parameter-Efficient Fine-Tuning on Multilingual ASR for Frisian (2024)
8. [^47^] Fine-tuning Strategies for ASR of Autism Spectrum Disorder Speech (Interspeech 2025)
9. [^48^] LoRA-Finetuned Whisper ASR (2024)
10. [^52^] Efficient Distillation of Multi-Task Speech Models via Language-Specific Experts (2024)
11. [^62^] Benchmarking Akan ASR Models Across Domain (2025)
12. SpecAugment: A Simple Data Augmentation Method for ASR (Park et al., 2019)
13. OpenAI Whisper tokenizer.py - github.com/openai/whisper
14. WaxalNLP Dataset - huggingface.co/datasets/google/WaxalNLP
15. [^29^] Winning Solution to the Swahili ASR Challenge - Zindi (2025)
