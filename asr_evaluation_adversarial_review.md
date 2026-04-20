# ADVERSARIAL REVIEW: ASR Training Notebook Evaluation Methodology
## Target: Whisper Medium Fine-tuning for Twi (Akan) ASR
### Review Date: 2025
### Focus: Evaluation Methodology, Metrics Computation, and Benchmarking Rigor

---

## EXECUTIVE SUMMARY

This notebook contains **CRITICAL** evaluation methodology flaws that render any reported WER numbers unreliable and incomparable with published benchmarks. The most severe issue is the use of the test set as the evaluation set during training with `load_best_model_at_end=True`, which constitutes direct data leakage and invalidates the entire evaluation. Additionally, WER is computed without text normalization, CER is absent, no per-domain evaluation is performed, and there is a language code mismatch in the tokenizer configuration. These issues collectively mean that **any reported WER < 10% would be scientifically meaningless** and cannot be compared against SOTA results.

**Severity Distribution:**
- CRITICAL: 2 issues
- HIGH: 3 issues
- MEDIUM: 2 issues
- LOW: 2 issues

---

---

## ISSUE 1: TEST SET USED FOR EVALUATION DURING TRAINING (DATA LEAKAGE)

### SEVERITY: CRITICAL

### Evidence (Notebook Lines 1474-1479):
```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=asr_data["train"],
    eval_dataset=asr_data["test"],   # <-- TEST SET USED AS EVAL SET
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)
```

Combined with (Notebook Lines 1437):
```python
    load_best_model_at_end=True,     # <-- Best model selected based on "test" set
    metric_for_best_model="wer",
    greater_is_better=False,
```

### Problem Analysis:

This is the single most damaging flaw in the entire pipeline. Here's the chain of failure:

1. **No validation split exists**: The notebook only creates `train` and `test` splits by merging the train/test splits from WaxalNLP, Farmerline, and Common Voice. There is **no held-out validation set**.

2. **Test set is used for evaluation during training**: The `eval_dataset` is set to `asr_data["test"]`. This means every 100 training steps, the model generates predictions on the test set and computes WER.

3. **Model selection uses test set performance**: With `load_best_model_at_end=True` and `metric_for_best_model="wer"`, the checkpoint with the lowest WER on the "evaluation" set (which is the TEST SET) is automatically loaded as the final model.

4. **This IS data leakage**: The test set, which should be used **only once** for final evaluation, is being used repeatedly during training to select the best model. The model is being optimized for test set performance through checkpoint selection.

### Impact:
- **Reported WER is optimistically biased**: The "best" model is selected specifically to minimize WER on what is supposed to be the unseen test set. This overestimates performance.
- **No true generalization estimate exists**: Because the test set influenced model selection, there is no unbiased estimate of how the model performs on truly unseen data.
- **Cannot be compared to benchmarks**: Published benchmarks evaluate on held-out test sets that were never used during training or model selection. This notebook violates that fundamental requirement.
- **Severity estimate**: The optimistic bias can range from 2-15 percentage points of WER depending on dataset size and eval frequency (every 100 steps with 3 saved checkpoints means up to 30 full evaluations on the test set).

### What the Literature Says:
The Akan ASR benchmarking paper (Azunre et al., 2025) explicitly states: "evaluations were performed using the original dataset splits provided by each source" and "All models were evaluated using the same test sets drawn from each dataset." This implies proper train/val/test separation was used for all published results.

### Corrected Approach:
See Section "RIGOROUS EVALUATION FRAMEWORK" below for the corrected split strategy.

---

---

## ISSUE 2: WER COMPUTED WITHOUT TEXT NORMALIZATION

### SEVERITY: CRITICAL

### Evidence (Notebook Lines 1381-1394):
```python
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

### Problem Analysis:

The `compute_metrics` function computes WER on **raw decoded strings** without any text normalization. The decoded predictions from Whisper are orthographic (they may contain casing variations, whitespace inconsistencies, and residual special tokens), while the references are the cleaned training labels decoded back from token IDs.

This creates multiple problems:

1. **Casing inconsistency**: The `clean_transcription()` function lowercases training text, so the model is trained on lowercase-only targets. However, Whisper's decoder may produce mixed-case output. Without lowercasing the predictions, every word with a capital letter counts as an error.

2. **Punctuation mismatch**: The training labels have had punctuation removed via `clean_transcription()`, but Whisper may generate punctuation. Each punctuation mark may be treated as a separate token or word, inflating the error count.

3. **Trailing whitespace artifact**: `clean_transcription()` adds `" "` (a trailing space) to every transcription. This means every reference ends with a space. If the prediction doesn't match this exactly, it creates a word-boundary mismatch.

4. **No standardization**: Standard ASR evaluation practices (used in MLPerf, OpenAI Whisper paper, and HuggingFace benchmarks) normalize both predictions and references before computing WER. This includes:
   - Lowercasing
   - Removing punctuation
   - Normalizing whitespace
   - Standardizing numbers (if applicable)

### Impact:
- **WER is artificially inflated**: Without normalization, the reported WER includes errors from formatting differences, not actual transcription errors. For a language like Twi with diacritics (ɛ, ɔ, ŋ) and potential casing issues, this can inflate WER by 5-20 percentage points.
- **Incomparable with benchmarks**: Published benchmarks for Akan ASR (Azunre et al., 2025) report WER after "converting the text to lowercase and removing punctuation and special characters." The notebook's raw WER cannot be directly compared.
- **Target WER < 10% is meaningless**: If the WER is computed without normalization, achieving "< 10%" would actually indicate excellent performance (since it's on harder, unnormalized text). But without knowing the normalization pipeline of comparison benchmarks, the target is ill-defined.

### What the Literature Says:

The HuggingFace Audio Course (chapter on evaluation) explicitly recommends:
> "We can train our systems on orthographic transcriptions, and then normalise the predictions and targets before computing the WER. This way, we train our systems to predict fully formatted text, but also benefit from the WER improvements we get by normalising the transcriptions."

The MLPerf Inference benchmark uses:
> "Text strings are reduced to lower-case and limited to alphabetic characters only. All punctuation is removed. OpenAI's EnglishTextNormalizer is used to control for regional spelling differences and word contractions."

The Akan ASR benchmarking paper (2025) states:
> "To match the reference transcripts and focus solely on the word content, predicted transcriptions were normalized by converting the text to lowercase and removing punctuation and special characters."

### Corrected Approach:
Use Whisper's `BasicTextNormalizer` (or a custom normalizer) on BOTH predictions and references before WER computation. See the corrected code in Section "RIGOROUS EVALUATION FRAMEWORK."

---

---

## ISSUE 3: MISSING CER (CHARACTER ERROR RATE)

### SEVERITY: HIGH

### Evidence:
The `compute_metrics` function only computes WER:
```python
    return {"wer": wer}
```
CER is never computed anywhere in the notebook.

### Problem Analysis:

CER (Character Error Rate) is an essential diagnostic metric for ASR evaluation, especially for low-resource languages with rich orthographies like Twi (Akan). CER provides character-level granularity that WER cannot capture. Without CER:

1. **Cannot distinguish phoneme errors from diacritic errors**: Twi uses special characters (ɛ, ɔ, ŋ) and tone markers. A model might get the phonemes right but miss diacritics (or vice versa). WER treats "kɔ" vs "ko" as a full word error, while CER would show it as a 1-character substitution out of many.

2. **Cannot assess near-misses**: A prediction of "mepɛ" instead of "mepa" is one character off but counts as a full word error in WER. CER would show this as minimal error.

3. **Cannot compare with published benchmarks**: The Akan ASR benchmarking paper (Azunre et al., 2025) reports BOTH WER and CER for all models across all datasets. Without CER, comparability is severely limited.

4. **Misses diagnostic value**: CER helps identify whether the model has learned the orthographic system. High WER with low CER suggests word segmentation issues; high WER with high CER suggests the model hasn't learned the phonetic system.

### Impact:
- **Incomplete evaluation picture**: Only WER provides a single coarse-grained number that can be misleading.
- **Cannot identify error patterns**: Without CER, you cannot tell if errors are at the phoneme level or the word level.
- **Cannot benchmark against SOTA**: Published results for Akan ASR (e.g., WER ~30% and CER ~12% on UGSpeechData for in-domain Whisper models) require both metrics.

### What the Literature Says:

The Akan ASR benchmarking paper reports paired WER/CER for all evaluations:
- Model 3 (Whisper-small, UGSpeechData-trained): WER ≈ 30%, CER ≈ 12%
- Model 1 (Whisper-small, Bible-trained): WER ≈ 35%, CER ≈ 11%
- Model 5 (wav2vec2, Financial Inclusion): WER ≈ 10%, CER ≈ 6% (in-domain)

These paired metrics are essential for understanding model behavior.

---

---

## ISSUE 4: NO PER-DOMAIN EVALUATION

### SEVERITY: HIGH

### Evidence:
The notebook merges three datasets (WaxalNLP, Farmerline, Common Voice) into a single `train` and `test` split:

```python
asr_data["train"] = interleave_datasets([asr_data["train"], farmerline["train"]], ...)
asr_data["test"] = interleave_datasets([asr_data["test"], farmerline["test"]], ...)
# Common Voice also merged similarly
```

The evaluation computes a single WER across all combined test data:
```python
trainer = Seq2SeqTrainer(
    ...
    eval_dataset=asr_data["test"],  # Combined from all sources
    compute_metrics=compute_metrics,  # Single WER number
)
```

### Problem Analysis:

The notebook trains on a combined dataset from three different sources:
1. **WaxalNLP aka_asr**: Google-sponsored Akan speech dataset
2. **Farmerline Twi Dataset 2.0**: Agricultural/financial domain speech
3. **Mozilla Common Voice Twi**: Crowdsourced read speech

Each dataset has different acoustic characteristics, vocabulary, and speaking styles:
- WaxalNLP: Potentially more diverse, possibly read speech
- Farmerline: Domain-specific (agricultural/financial terminology)
- Common Voice: Short, crowdsourced utterances with acoustic variability

The benchmarking literature (Azunre et al., 2025) demonstrates that **domain mismatch is the single biggest challenge for Akan ASR**. Models that perform well on one dataset often fail catastrophically on another:
- Models trained on UGSpeechData: WER ~30% in-domain, WER >70% out-of-domain
- Models trained on Financial Inclusion: WER ~10% in-domain, WER >86% out-of-domain
- Whisper models even exhibit "decoder collapse" (WER >100%) on mismatched domains

By merging all datasets and reporting a single WER:
1. **Cannot identify dataset-specific issues**: If the model fails on Farmerline but excels on WaxalNLP, this is invisible.
2. **Cannot assess domain generalization**: The primary challenge in Akan ASR is cross-domain robustness, which cannot be measured with a single aggregate number.
3. **Cannot compare with published per-domain results**: Published benchmarks report per-dataset WER/CER.

### Impact:
- **A single WER is misleading**: A WER of 25% could mean 15% on WaxalNLP, 20% on Common Voice, and 40% on Farmerline -- or any other combination. The number is uninterpretable.
- **Domain-specific failure modes are hidden**: The model might perform catastrophically on the financial domain (which may be the most important use case) while looking good on aggregate.

---

---

## ISSUE 5: TOKENIZER LANGUAGE CODE MISMATCH

### SEVERITY: HIGH

### Evidence (Notebook Lines 961 and 944):
```python
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="yoruba", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="yoruba", task="transcribe")
```

And (Notebook Line 1229):
```python
model.generation_config.language = "yoruba"
```

### Problem Analysis:

The notebook fine-tunes a model for **Twi** (Akan) speech recognition but configures the tokenizer and generation with `language="yoruba"`. This is fundamentally incorrect:

1. **Language mismatch at tokenizer level**: Whisper uses language-specific token prefixes. When `language="yoruba"` is set, the tokenizer preparts Yoruba-specific tokens. Twi (Akan) and Yoruba are distinct languages from different language families (Niger-Congo but different branches: Kwa vs. Volta-Niger).

2. **Forced decoder bias**: Setting `model.generation_config.language = "yoruba"` forces the decoder to generate text as if it were Yoruba, not Twi. While the model can learn to override this through fine-tuning, it introduces an unnecessary language modeling bias that may:
   - Favor Yoruba phoneme patterns over Twi
   - Affect the tokenizer's BPE segmentation
   - Introduce Yoruba-specific tokens into the vocabulary

3. **Model card inconsistency**: The kwargs claim `language="tw"` (Twi), but the actual code uses Yoruba.

### Impact:
- **Suboptimal performance**: The model is fighting against a Yoruba language bias during training and inference. This could increase WER by an unknown amount.
- **Incorrect language metadata**: The model will report Yoruba as its language, which affects downstream applications and benchmarking.
- **Cannot fairly compare**: A model configured for Yoruba but evaluated on Twi is not a fair comparison with models properly configured for Twi/Akan.

### Corrected Approach:
Use the correct language identifier for Twi/Akan. Whisper's language codes include:
- `"akan"` or `"tw"` for Twi (Akan)
- Check Whisper's language list for the exact code

The tokenizer should be initialized as:
```python
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="akan", task="transcribe")
```

---

---

## ISSUE 6: TRAIN/EVAL MISMATCH IN TRANSCRIPTION CLEANING

### SEVERITY: MEDIUM

### Evidence (Notebook Lines 1103-1118):
```python
def clean_transcription(text: str) -> str:
    return re.sub(CHARS_TO_IGNORE_REGEX, '', text).lower().strip() + " "

def prepare_dataset(batch):
    ...
    batch["labels"] = tokenizer(clean_transcription(batch["transcription"])).input_ids
```

And the `compute_metrics` function decodes label_ids back directly:
```python
label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
```

### Problem Analysis:

During training, transcriptions are cleaned (lowercased, punctuation stripped) before tokenization. During evaluation, the `label_ids` are decoded back to strings. Since the labels were tokenized from cleaned text, the decoded references will also be cleaned text. This is actually **consistent** -- both training targets and evaluation references use the same cleaned format.

However, there is a subtle issue:

1. **The trailing space artifact**: `clean_transcription()` adds `" "` (a trailing space) to every transcription. This means the reference strings end with a space. When decoded, these labels retain the trailing space pattern from tokenization.

2. **The model learns to predict trailing spaces**: Since every training target ends with a space, the model learns to generate a trailing space. This is an artifact of the cleaning function, not a linguistic feature.

3. **WER impact**: If the model sometimes fails to generate the trailing space (which is likely, as it's a spurious pattern), each missing trailing space could count as a word-boundary error.

### Impact:
- **Minor WER inflation**: The trailing space artifact may cause a small number of boundary errors.
- **The cleaning is consistent train/eval**: This is actually correct -- the model is evaluated on the same type of text it was trained on. The issue is that the cleaning itself is non-standard and not documented.

---

---

## ISSUE 7: NO USE OF WHISPER'S BUILT-IN NORMALIZATION

### SEVERITY: MEDIUM

### Problem Analysis:

Whisper was released with a `BasicTextNormalizer` that standardizes text before WER computation. The notebook does not use this. The custom regex approach (`CHARS_TO_IGNORE_REGEX`) is inferior because:

1. **Limited coverage**: The regex only removes: `, ? . ! - ; : " % ' \`. It does not handle:
   - Other punctuation (e.g., parentheses, brackets, slashes)
   - Number normalization (if numbers exist in Twi transcripts)
   - Whitespace normalization (multiple spaces, tabs)
   - Unicode normalization (for Twi's special characters)

2. **No standardization**: `BasicTextNormalizer` also handles:
   - Removing non-word characters systematically
   - Stripping extra whitespace
   - Lowercasing
   - Handling special cases

3. **Community standard**: Most Whisper fine-tuning benchmarks (including the official HuggingFace tutorial and MLPerf) use the built-in normalizer for comparability.

### Impact:
- **Incomparable with Whisper-based benchmarks**: Different normalization pipelines produce different WER numbers.
- **Inconsistent error counting**: Unhandled punctuation and formatting differences inflate the error count artificially.

---

---

## ISSUE 8: MODEL CARD DISCREPANCIES

### SEVERITY: LOW

### Evidence (Notebook Lines 1589-1600):
```python
kwargs = {
    "dataset_tags": ["google/WaxalNLP", "ghananlpcommunity/twi_dataset_2.0_farmerline"],
    "dataset": "WaxalNLP aka_asr, Twi Dataset 2.0 Farmerline",
    "language": "tw",
    "model_name": "Whisper Medium SerendepifyLabs Twi ASR",
    "finetuned_from": "openai/whisper-small",  # <-- WRONG: actually whisper-medium
    "tasks": "automatic-speech-recognition",
}
```

### Problem Analysis:

1. **`finetuned_from` is incorrect**: The code loads `openai/whisper-medium` (Line 464), but the model card claims it was fine-tuned from `openai/whisper-small`. This is factually wrong and misleads users about the model's size and capabilities.

2. **Common Voice not mentioned**: The model card only lists WaxalNLP and Farmerline, but the code also loads Mozilla Common Voice data. The dataset list is incomplete.

3. **Language code inconsistency**: The model card says `language="tw"` but the code uses `language="yoruba"`.

### Impact:
- **Misleading metadata**: Users downloading the model will have incorrect information about its origin and training data.
- **Reproducibility issues**: Attempting to reproduce the training with `whisper-small` instead of `whisper-medium` would yield different results.

---

---

## ISSUE 9: NO STATISTICAL SIGNIFICANCE TESTING

### SEVERITY: LOW

### Problem Analysis:

The notebook reports a single WER number without any measure of uncertainty:
- No confidence intervals
- No standard deviation across multiple evaluation runs
- No bootstrap resampling for WER variance estimation
- No paired t-test for comparing with baseline models

The Akan ASR benchmarking paper reports "WER and CER were calculated for each dataset-model combination, accompanied by the standard deviation and 95% confidence intervals." Without statistical measures, it is impossible to know whether a reported WER of 9.8% is meaningfully different from 10.2%.

### Impact:
- **Cannot claim WER < 10% with confidence**: A single WER of 9.9% might not be statistically distinguishable from 10.1%.
- **No rigorous comparison possible**: Without confidence intervals, comparing against SOTA is meaningless.

---

---

## SOTA CONTEXT FOR TWI (AKAN) ASR

Based on the most comprehensive Akan ASR benchmarking study to date (Azunre et al., 2025), here are the current SOTA results:

### Per-Dataset Performance (Published Benchmarks):

| Model | Architecture | Training Data | UGSpeechData WER | Common Voice WER | Bible WER | Financial Inclusion WER |
|-------|-------------|---------------|-----------------|------------------|-----------|------------------------|
| Model 1 | Whisper-small | Bible (Lagyamfi) | ~80% | ~64% | **~37%** | N/A |
| Model 2 | Whisper-large | UGSpeechData | N/A | N/A | N/A | >100% (decoder collapse) |
| Model 3 | Whisper-small | UGSpeechData | **~30%** | ~77% | ~78% | >100% |
| Model 5 | wav2vec2-xls-r-300m | Financial Inclusion | >90% | ~98% | ~95% | **~10%** |
| Model 6 | wav2vec2-xls-r-300m | UGSpeechData | ~31% | ~69% | ~70% | ~86% |

### Key Observations:
1. **The best in-domain WER is ~10%** (Model 5 on Financial Inclusion)
2. **Cross-domain performance is catastrophic**: Models routinely achieve WER >70% out-of-domain
3. **Whisper models show decoder collapse** on domain-mismatched financial data
4. **The best UGSpeechData WER is ~30%**
5. **No single model achieves WER < 10% across all domains**

### Implications for the Target "WER < 10%":
- Achieving WER < 10% on ALL combined test data would be **unprecedented** for Akan ASR
- If the claim is WER < 10% on normalized text, this must be clearly stated
- The only published result near this level is in-domain evaluation on Financial Inclusion (~10%)
- Without proper normalization, data splits, and per-domain reporting, a WER < 10% claim would not be credible

---

---

## RIGOROUS EVALUATION FRAMEWORK (CORRECTED)

Below is the corrected evaluation methodology with proper data splits, normalization, and comprehensive metrics.

### 1. Proper Data Split Strategy

```python
from datasets import load_dataset, DatasetDict, Audio, interleave_datasets

SAMPLING_RATE = 16000

# --- Load individual datasets with their ORIGINAL splits ---
# 1. WaxalNLP
waxal_train = load_dataset("google/WaxalNLP", "aka_asr", split="train", streaming=True)
waxal_test = load_dataset("google/WaxalNLP", "aka_asr", split="test", streaming=True)

# 2. Farmerline
farmerline = load_dataset("ghananlpcommunity/twi_dataset_2.0_farmerline", streaming=True)

# 3. Common Voice (load via Mozilla API as in notebook)
# ... (same loading code) ...

# --- Strategy: Create TRAIN / VALIDATION / TEST splits ---
# The ORIGINAL test splits from each dataset become the FINAL test set (NEVER touched during training)
# We create a VALIDATION split from the training data for model selection

# For combined training (use only train portions):
combined_train = interleave_datasets(
    [waxal_train, farmerline["train"], common_voice_train],
    stopping_strategy="all_exhausted",
    seed=42
).shuffle(seed=42, buffer_size=1000)

# Create VALIDATION split from training data (e.g., 5-10%)
# For streaming datasets, use take/skip:
# val_size = 500
# val_data = combined_train.take(val_size)
# train_data = combined_train.skip(val_size)

# FINAL TEST SET: Keep each dataset's test split SEPARATE for per-domain evaluation
test_sets = {
    "waxal": waxal_test,
    "farmerline": farmerline["test"],
    "common_voice": common_voice_test,
}
```

### 2. Corrected compute_metrics with Normalization

```python
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Use Whisper's built-in normalizer
normalizer = BasicTextNormalizer()

def normalize_text(text: str) -> str:
    """Normalize text for fair WER/CER computation."""
    normalized = normalizer(text)
    return normalized.strip()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # --- NORMALIZE BOTH PREDICTIONS AND REFERENCES ---
    pred_str_norm = [normalize_text(s) for s in pred_str]
    label_str_norm = [normalize_text(s) for s in label_str]

    # Filter out empty references to avoid division by zero
    filtered_preds = []
    filtered_labels = []
    for p, l in zip(pred_str_norm, label_str_norm):
        if len(l) > 0:
            filtered_preds.append(p)
            filtered_labels.append(l)

    # Compute WER on normalized text
    wer = 100 * wer_metric.compute(
        predictions=filtered_preds,
        references=filtered_labels
    )

    # Compute CER on normalized text
    cer = 100 * cer_metric.compute(
        predictions=filtered_preds,
        references=filtered_labels
    )

    return {"wer": wer, "cer": cer}
```

### 3. Per-Domain Evaluation Function

```python
def evaluate_per_domain(model, processor, test_sets, tokenizer):
    """
    Evaluate model on each dataset domain separately.
    """
    from transformers import pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    all_results = {}
    all_predictions = []
    all_references = []

    for domain_name, dataset in test_sets.items():
        predictions = []
        references = []

        for sample in dataset:
            audio = sample["audio"]
            reference = sample["transcription"]

            result = pipe(audio["array"], return_timestamps=False)
            prediction = result["text"]

            predictions.append(normalizer(prediction))
            references.append(normalizer(reference))

        # Filter empty references
        filtered_preds = [p for p, r in zip(predictions, references) if len(r) > 0]
        filtered_refs = [r for r in references if len(r) > 0]

        domain_wer = 100 * wer_metric.compute(
            predictions=filtered_preds, references=filtered_refs
        )
        domain_cer = 100 * cer_metric.compute(
            predictions=filtered_preds, references=filtered_refs
        )

        all_results[domain_name] = {
            "wer": domain_wer,
            "cer": domain_cer,
            "n_samples": len(filtered_refs),
        }

        all_predictions.extend(filtered_preds)
        all_references.extend(filtered_refs)

    # Aggregate across all domains
    all_results["aggregate"] = {
        "wer": 100 * wer_metric.compute(
            predictions=all_predictions, references=all_references
        ),
        "cer": 100 * cer_metric.compute(
            predictions=all_predictions, references=all_references
        ),
        "n_samples": len(all_references),
    }

    return all_results
```

### 4. Corrected Trainer Setup

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_medium-twi-asr",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    optim="adamw_torch",
    # --- EVALUATION ON VALIDATION SET, NOT TEST SET ---
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    logging_steps=25,
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=225,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard"],
    dataloader_num_workers=4,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,        # Training data only
    eval_dataset=val_data,           # VALIDATION set (from train split)
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Includes WER + CER + normalization
    processing_class=processor.feature_extractor,
)

# Train (model selection uses VALIDATION set)
trainer.train()

# --- FINAL EVALUATION: Run ONCE on held-out TEST set ---
final_results = evaluate_per_domain(
    model=trainer.model,
    processor=processor,
    test_sets=test_sets,
    tokenizer=tokenizer,
)

print("=" * 60)
print("FINAL TEST SET RESULTS (REPORT THESE)")
print("=" * 60)
for domain, metrics in final_results.items():
    print(f"{domain:15s}: WER={metrics['wer']:.2f}%  CER={metrics['cer']:.2f}%  (n={metrics['n_samples']})")
```

### 5. Statistical Confidence Intervals

```python
import numpy as np

def compute_wer_with_confidence(predictions, references, confidence=0.95, n_bootstrap=1000):
    """Compute WER with bootstrap confidence intervals."""
    wer_metric = evaluate.load("wer")

    # Pairwise WER (per-sample)
    sample_wers = []
    for pred, ref in zip(predictions, references):
        if len(ref) > 0:
            sample_wers.append(wer_metric.compute(predictions=[pred], references=[ref]))

    overall_wer = wer_metric.compute(predictions=predictions, references=references)

    # Bootstrap confidence interval
    bootstrap_wers = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(sample_wers), size=len(sample_wers), replace=True)
        bootstrap_wers.append(np.mean([sample_wers[i] for i in indices]))

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_wers, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_wers, 100 * (1 - alpha / 2))

    return {
        "wer": 100 * overall_wer,
        "ci_lower": 100 * ci_lower,
        "ci_upper": 100 * ci_upper,
        "std": 100 * np.std(sample_wers),
    }
```

### 6. Corrected Model Card kwargs

```python
kwargs = {
    "dataset_tags": [
        "google/WaxalNLP",
        "ghananlpcommunity/twi_dataset_2.0_farmerline",
        "mozilla-foundation/common_voice_11_0",
    ],
    "dataset": "WaxalNLP aka_asr, Twi Dataset 2.0 Farmerline, Common Voice 11 Twi",
    "language": "tw",
    "model_name": "Whisper Medium SerendepifyLabs Twi ASR",
    "finetuned_from": "openai/whisper-medium",  # CORRECTED
    "tasks": "automatic-speech-recognition",
    "eval_metrics": "wer, cer",
}
```

### 7. Corrected Tokenizer Initialization

```python
# Use Akan/Twi language code, NOT Yoruba
tokenizer = WhisperTokenizer.from_pretrained(
    model_id,
    language="akan",  # CORRECTED: was "yoruba"
    task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    model_id,
    language="akan",  # CORRECTED: was "yoruba"
    task="transcribe"
)
model.generation_config.language = "akan"  # CORRECTED
```

---

---

## SUMMARY OF ISSUES AND IMPACT

| # | Issue | Severity | Impact on WER | Impact on Benchmarking |
|---|-------|----------|---------------|----------------------|
| 1 | Test set used for eval + model selection | CRITICAL | Optimistically biased by 2-15pp | Results completely incomparable |
| 2 | WER without normalization | CRITICAL | Inflated by 5-20pp (direction uncertain) | Cannot compare with normalized benchmarks |
| 3 | Missing CER | HIGH | N/A (missing diagnostic) | Cannot compare with SOTA papers |
| 4 | No per-domain evaluation | HIGH | Unknown bias direction | Single number is uninterpretable |
| 5 | Tokenizer language="yoruba" | HIGH | Unknown degradation | Model is for wrong language |
| 6 | Trailing space artifact | MEDIUM | Minor inflation | Low impact |
| 7 | No Whisper normalizer | MEDIUM | Inconsistent with Whisper benchmarks | Low comparability |
| 8 | Model card lies about base model | LOW | None | Reproducibility issue |
| 9 | No statistical testing | LOW | None | Cannot claim significance |

### Cumulative Impact:

The combination of Issues 1, 2, 4, and 5 means that:

1. **Any reported WER is unreliable** due to test-set leakage during model selection
2. **The WER number cannot be compared** with published benchmarks due to missing normalization
3. **The single aggregate WER hides** domain-specific failures
4. **The wrong language code** may be causing systematic errors not captured by WER

### Bottom Line:

**This notebook cannot produce a scientifically valid WER estimate for Twi ASR.** The evaluation pipeline must be completely rewritten following the corrected framework above before any claims about WER performance (including the target of WER < 10%) can be considered credible.

---

---

## RECOMMENDATIONS

### Immediate Actions Required:

1. **Implement proper train/validation/test splits**: Create a validation set from training data. Use test sets ONLY for final evaluation.

2. **Add text normalization to compute_metrics**: Use Whisper's `BasicTextNormalizer` on both predictions and references before WER computation.

3. **Add CER**: Include Character Error Rate in the metrics.

4. **Fix the language code**: Change from "yoruba" to "akan" or "tw".

5. **Implement per-domain evaluation**: Report WER/CER separately for each dataset source.

6. **Fix the model card**: Correct `finetuned_from` to `openai/whisper-medium`.

7. **Run final evaluation only once**: After training completes, evaluate on each test set exactly once.

8. **Add confidence intervals**: Use bootstrap resampling to report 95% CIs for all metrics.

### Benchmarking Protocol:

To claim WER < 10% (or any target), the following must be demonstrated:

1. **Normalized WER** (using the same normalization as comparison benchmarks)
2. **Per-domain WER/CER** for each dataset source
3. **Held-out test set** never used during training or model selection
4. **Confidence intervals** showing statistical significance
5. **Comparison with published baselines** on the same datasets with the same protocol
6. **Qualitative error analysis** showing typical error patterns

---

*Review compiled based on notebook analysis, Akan ASR benchmarking literature (Azunre et al., 2025), HuggingFace Whisper fine-tuning best practices, and MLPerf ASR evaluation standards.*
