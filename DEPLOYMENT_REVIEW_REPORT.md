
# ADVERSARIAL CLOUD DEPLOYMENT READINESS REVIEW
## ASR Training Notebook: Whisper Fine-tuning for Twi (Akan) ASR

**Notebook Analyzed**: `/mnt/agents/notebook.ipynb`
**Model**: `openai/whisper-medium` (769M parameters, ~5GB inference VRAM)
**Target Languages**: Twi (Akan) - `tw` / `aka`
**Datasets**: WaxalNLP (aka_asr), Farmerline (twi_dataset_2.0), Common Voice (Twi scripted)
**Platforms Evaluated**: RunPod, Modal.com, Unsloth, Vast.ai, Lambda Labs

---

## EXECUTIVE SUMMARY

This notebook has **SEVERE deployment blockers** for any cloud GPU platform. It was written
for Google Colab/Kaggle and contains patterns fundamentally incompatible with reproducible,
fault-tolerant cloud training. The notebook title claims "whisper small" but the code loads
`openai/whisper-medium` (a 3x larger model). There are critical language mismatches,
failed cells, no checkpoint resumption, no containerization, no dependency pinning, and
no fault tolerance. **Training on cloud platforms without significant modifications will
result in wasted compute and likely failures.**

---

## CRITICAL ISSUES (Will Cause Training Failure)

### C1. MODEL NAME INCONSISTENCY: Title Says "Small", Code Uses "Medium"
**Severity**: CRITICAL
- **Title/comment**: "fine-tunes whisper small model for Twi transcription"
- **Actual code**: `model_id = "openai/whisper-medium"` (769M params)
- **output_dir**: `"./whisper_small-waxal-farmerline_akan-asr"` (claims small)
- **finetuned_from**: `"openai/whisper-small"` (incorrect metadata)
- **VRAM Impact**: Medium requires ~5GB inference VRAM vs ~2GB for small. Training
  needs 15-25GB depending on batch size — this changes GPU selection entirely.
- **Fix**: Align all references. Use `model_id = "openai/whisper-small"` if the 
  small model is intentional, or update all labels to "medium".

### C2. NOTEBOOK CELL ACTUALLY FAILED DURING EXECUTION
**Severity**: CRITICAL
- Cell 13 (execution_count: 13) has `"status": "failed"` in papermill metadata
- This means the notebook was NOT successfully run end-to-end even in its original environment
- Deploying unverified code to paid cloud GPU = guaranteed wasted money
- **Fix**: Verify all cells execute successfully before any cloud deployment.

### C3. LANGUAGE MISMATCH: Tokenizer Configured for Yoruba, Not Twi/Akan
**Severity**: CRITICAL
- `tokenizer = WhisperTokenizer.from_pretrained(model_id, language="yoruba", task="transcribe")`
- `model.generation_config.language = "yoruba"`
- The notebook title says "Twi" and the dataset is Akan/Twi (`language="tw"`)
- Yoruba (`yo`) and Twi (`tw`) are different languages with different character sets
- This will produce **garbled output** and the model will learn wrong language tokens
- **Fix**: Use `language="twi"` or `language="akan"` — verify Whisper's supported language codes

### C4. NO REQUIREMENTS.TXT / NO DOCKERFILE / NO ENVIRONMENT SPEC
**Severity**: CRITICAL
- Notebook uses `!pip install` commands with specific versions: `transformers==4.52.0` (YANKED),
  `datasets==3.6.0`, `soundfile>=0.12.1`
- Version 4.52.0 of transformers is **YANKED** from PyPI (confirmed in notebook output)
- No `requirements.txt`, `pyproject.toml`, `environment.yml`, or Dockerfile exists
- Cloud platforms build containers from spec files — this notebook cannot be reproduced
- **Fix**: Create a pinned `requirements.txt` with verified working versions

### C5. NO CHECKPOINT RESUMPTION ON PREEMPTION
**Severity**: CRITICAL
- `trainer.train()` starts from scratch every time
- Cloud spot/preemptible instances can be interrupted at any time
- 1000 steps with no resumption = complete loss of progress on interruption
- No `resume_from_checkpoint` parameter in TrainingArguments
- **Fix**: Add `resume_from_checkpoint=True` and persist checkpoints to Network Volume

---

## HIGH SEVERITY ISSUES (Major Problems)

### H1. STREAMING DATASETS WITH NO CACHING
**Severity**: HIGH
- Uses `load_dataset(..., streaming=True)` for WaxalNLP and Farmerline
- Common Voice data is downloaded via API call + `!wget` each run
- On every training restart, ALL data must be re-downloaded from the internet
- HuggingFace Hub streaming is unreliable for long training runs (connection drops,
  rate limits, temporary outages)
- **Fix**: Download datasets fully to persistent storage before training starts

### H2. NO ENVIRONMENT VARIABLE VALIDATION
**Severity**: HIGH
- `hf_token = os.environ.get("HF_TOKEN")` — no validation that it's set
- `mozilla_key = os.environ.get("MOZILLA_APIKEY")` — no validation
- If HF_TOKEN is missing: cryptic auth errors hours into training
- If MOZILLA_APIKEY is missing: API call fails, Common Voice data silently missing
- **Fix**: Add validation with clear error messages at startup

### H3. NO EXPERIMENT TRACKING (TensorBoard Only)
**Severity**: HIGH
- `report_to=["tensorboard"]` — TensorBoard logs are local only by default
- No Weights & Biases (W&B) integration for cloud experiment tracking
- No way to monitor training remotely or compare runs
- On pod interruption, TensorBoard logs may be lost
- **Fix**: Add W&B integration: `report_to=["tensorboard", "wandb"]`

### H4. DATA LOADER WORKER MISMATCH WITH STREAMING
**Severity**: HIGH
- `dataloader_num_workers=4` but streaming datasets have only 1-2 shards
- Trainer emits warnings: "Too many dataloader workers: 4 (max is dataset.num_shards=2)"
- Wasted processes, CPU overhead, no performance benefit
- **Fix**: Set `dataloader_num_workers` based on actual dataset shard count

### H5. NO ERROR HANDLING FOR EXTERNAL API CALLS
**Severity**: HIGH
- Common Voice download uses raw `requests.post()` with no retry logic
- `response.json().get("downloadUrl")` — no check for failed auth or rate limiting
- `!wget` command has no retry, no timeout, no checksum verification
- Network hiccup = training fails with uninformative error
- **Fix**: Wrap API calls in retry logic with exponential backoff

### H6. NO PERSISTENT STORAGE CONFIGURATION
**Severity**: HIGH
- `output_dir="./whisper_small-waxal-farmerline_akan-asr"` — local relative path
- On RunPod/Modal, container storage is ephemeral by default
- Checkpoints, logs, and model outputs lost on pod termination
- **Fix**: Mount persistent Network Volume and write all outputs there

### H7. GRADIO DEMO CANNOT RUN IN CLOUD
**Severity**: HIGH
- `iface.launch()` binds to localhost only — inaccessible from outside container
- No `share=True`, no `server_name="0.0.0.0"`, no port configuration
- Blocks the training process (no async/deployment pattern)
- **Fix**: Deploy demo separately as an inference endpoint, not part of training script

### H8. NO LOGGING CONFIGURATION
**Severity**: HIGH
- No structured logging — only print statements
- No log file persistence
- On remote cloud instances, logs are the ONLY debugging tool
- **Fix**: Add Python logging module with file and stdout handlers

---

## MEDIUM SEVERITY ISSUES

### M1. TRANSFORMERS VERSION IS YANKED
**Severity**: MEDIUM
- `transformers==4.52.0` was yanked from PyPI (visible in notebook output)
- Yanked packages may have critical bugs or security issues
- **Fix**: Pin to a stable, non-yanked version (e.g., `transformers==4.46.3` or `>=4.47,<4.53`)

### M2. NO MIXED PRECISION SAFETY CHECK
**Severity**: MEDIUM
- `fp16=torch.cuda.is_available()` — enables fp16 on ALL GPUs
- Some older GPUs don't support fp16 (e.g., V100 does, but some older cards don't)
- No check for `torch.cuda.is_bf16_supported()` (bf16 is preferred on Ampere+)
- **Fix**: Auto-detect optimal dtype: bf16 if available, else fp16 with fallback

### M3. NO TORCH.COMPILE OR FLASH ATTENTION
**Severity**: MEDIUM
- PyTorch 2.x's `torch.compile()` gives 4.5x speedup for Whisper forward pass
- Flash Attention 2 reduces memory usage significantly
- Neither is used — training is slower and uses more memory than necessary
- **Fix**: Add `torch.compile()` and Flash Attention 2 when supported

### M4. METADATA INCONSISTENCIES
**Severity**: MEDIUM
- `finetuned_from` says `"openai/whisper-small"` but model is medium
- `language` in kwargs says `"tw"` but tokenizer uses `"yoruba"`
- `model_name` says "Whisper Medium" but code references small
- **Fix**: Audit all metadata fields for consistency

### M5. NO NOTIFICATION ON COMPLETION/FAILURE
**Severity**: MEDIUM
- Training can take hours — user has no way to know if it completed or crashed
- No webhook, email, Slack, or W&B alert configured
- **Fix**: Add W&B alerts or a simple webhook notification

### M6. HARD-CODED OUTPUT DIRECTORY
**Severity**: MEDIUM
- Output dir is hardcoded: `"./whisper_small-waxal-farmerline_akan-asr"`
- No timestamp or run ID to distinguish between training runs
- Overwrites previous checkpoints if re-run
- **Fix**: Use parameterized output dir with timestamp/run ID

---

## LOW SEVERITY ISSUES

### L1. `trust_remote_code=True` in inference section without justification
### L2. Unused commented-out pip install cells at end of notebook
### L3. No `seed` set in TrainingArguments for reproducibility
### L4. No validation that `push_to_hub` succeeded
### L5. No cleanup of downloaded tarball after extraction

---

## VRAM & GPU SIZING ANALYSIS

### Whisper-Medium Training Memory Requirements

| Component | Estimated VRAM (fp16) |
|-----------|----------------------|
| Model weights (769M params, fp16) | ~1.5 GB |
| Gradients (fp16) | ~1.5 GB |
| Optimizer states (AdamW, fp32) | ~3.0 GB |
| Activations (batch=16, grad_checkpointing) | ~8-12 GB |
| Feature extractor overhead | ~1 GB |
| **Total estimated** | **~15-19 GB** |

### Recommended GPU Types

| GPU | VRAM | RunPod Cost/hr | Suitability |
|-----|------|----------------|-------------|
| RTX 4090 | 24GB | $0.34 | Marginal — tight fit, risk of OOM |
| A100 40GB | 40GB | $1.19 | **Recommended** — good fit with headroom |
| A100 80GB | 80GB | $1.39 | Excellent — comfortable, allows batch tuning |
| H100 80GB | 80GB | $2.69 | Overkill — premium price, no training benefit |
| L40S 48GB | 48GB | $0.79 | Good alternative — cost-effective |
| RTX 3090 | 24GB | $0.22 | Too risky — likely OOM with batch=16 |

**Recommendation**: Use A100 40GB or L40S 48GB for cost-optimal training.

### Training Duration & Cost Estimate

- From notebook output: ~0.88 steps/sec at observed throughput
- 1000 steps ≈ 18-20 minutes of actual training time
- With data loading, evaluation, checkpointing: ~30-45 minutes wall-clock
- **Estimated cost per run**: $0.60-$1.80 (A100 40GB to H100)
- With spot/preemptible pricing (60% off): $0.24-$0.72 per run

---

## PLATFORM-SPECIFIC DEPLOYMENT ANALYSIS

### RUNPOD DEPLOYMENT

**Issues**:
- Notebook uses `!apt-get` and `!pip` — RunPod expects pre-built container images
- No Network Volume configuration for checkpoint persistence
- No template/worker type specification
- Data downloaded at runtime via unreliable methods

**RunPod-Specific Recommendations**:
1. Use a Docker image with all dependencies pre-installed
2. Mount a Network Volume to `/workspace` for persistent storage
3. Use an A100 40GB pod (minimum) or L40S for cost efficiency
4. Enable checkpoint resumption with volume-persisted checkpoints
5. Use Spot instances for 60% cost savings with checkpoint fallback

### MODAL.COM DEPLOYMENT

**Issues**:
- Modal uses `@app.function()` decorator pattern — none present
- Modal builds containers from Image definitions — `!pip install` doesn't work
- No Modal Stub definition, no GPU spec, no volume mounting
- Gradio `iface.launch()` incompatible with Modal's serverless model

**Modal-Specific Recommendations**:
1. Define a Modal Stub with custom Image containing all dependencies
2. Use Modal Volume for dataset caching (download once, reuse across runs)
3. Deploy training as a Modal function with GPU spec
4. Deploy Gradio demo separately as a Modal web endpoint
5. Use Modal's `torch()` image builder for optimized PyTorch builds

### UNSLOTH COMPATIBILITY

**Issues**:
- Unsloth primarily optimizes decoder-only LLMs (Llama, Mistral, etc.)
- Whisper is an **encoder-decoder (seq2seq)** architecture
- Unsloth has model cards for Whisper (`unsloth/whisper-small`, `unsloth/whisper-large-v3`)
  but the optimization focus is on text LLMs, not speech models
- Unsloth does NOT support seq2seq training through its standard API

**Unsloth Verdict**: NOT RECOMMENDED for this use case.
- Unsloth's optimizations (gradient checkpointing fusion, manual autograd,
  4x faster training) are designed for decoder-only causal LM architectures
- Whisper's encoder-decoder architecture with cross-attention is fundamentally different
- The custom CUDA kernels Unsloth uses are not applicable to Whisper's forward pass

**Alternatives**:
- Use standard Transformers with `torch.compile()` for ~4.5x speedup
- Apply LoRA/QLoRA via PEFT for memory efficiency (see recommendations below)
- Use Flash Attention 2 for memory reduction

---

## COST OPTIMIZATION STRATEGIES

### 1. Switch to Whisper-Small (if accuracy allows)
- Whisper-small: 244M params vs medium's 769M
- VRAM drops from ~15-19GB to ~6-10GB for training
- Enables RTX 4090 (24GB) at $0.34/hr instead of A100 at $1.19/hr
- **Cost reduction: ~70% per training hour**
- For low-resource languages like Twi, small may generalize adequately

### 2. Use LoRA/QLoRA Instead of Full Fine-tuning
- LoRA reduces trainable parameters by ~90-95%
- VRAM reduction allows smaller/cheaper GPUs
- Training speed improves due to fewer gradients to compute
- **Recommended config**: LoRA rank=32, target_modules=["q_proj","v_proj","k_proj","o_proj"]

### 3. Use Spot/Preemptible Instances
- RunPod Spot: 60% cheaper than on-demand
- With checkpoint resumption, interruptions lose only partial progress
- For 1000-step training (30-45 min), interruption probability is low

### 4. Enable torch.compile()
- PyTorch 2.x compile gives 4.5x forward pass speedup for Whisper
- Reduces training time from ~45 min to ~10-15 min
- **Cost reduction: ~70% on time-based billing**

### 5. Use bf16 Instead of fp16
- bf16 has better numerical stability than fp16
- No need for gradient scaling
- Slightly faster on Ampere/Hopper GPUs

---

## RECOMMENDED TRAINING CONFIGURATION

### Optimized TrainingArguments

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=os.environ.get("OUTPUT_DIR", "./outputs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_bf16_supported(),  # bf16 preferred
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    tf32=True if torch.cuda.is_available() else False,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    optim="adamw_torch",
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=5,  # Increased for safety
    logging_steps=25,
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=225,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard", "wandb"],
    run_name=f"whisper-medium-twi-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    dataloader_num_workers=2,  # Match dataset shards
    dataloader_pin_memory=True,
    push_to_hub=True,
    hub_strategy="checkpoint",  # Push checkpoints to hub
    seed=42,
    data_seed=42,
    # Fault tolerance
    dataloader_drop_last=False,
    ignore_skip_data=False,
)
```

### LoRA Configuration (Alternative to Full Fine-tuning)

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

---

## FILES GENERATED

The following deployment-ready files have been generated in `/mnt/agents/output/`:

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image with all dependencies pre-installed |
| `requirements.txt` | Pinned Python dependencies |
| `runpod_template.yaml` | RunPod pod deployment template |
| `modal_train.py` | Modal.com training deployment script |
| `train_script.py` | Production-ready standalone training script |
| `check_env.py` | Environment validation script |
