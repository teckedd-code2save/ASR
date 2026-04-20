# ReachyAI Akan/Twi ASR Training Pipeline

> **Optimized ASR training for Akan (Twi) with multi-model support, aggressive caching, pseudo-labeling, and KenLM fusion. Built for [ReachyAI](https://github.com/teckedd-code2save/reachy-health-server) healthcare voice platform.**

---

## TL;DR: Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download Common Voice Twi from Mozilla Data Collective
#    Place extracted corpus at: ./data/cv-corpus-24.0-2025-12-05/tw/

# 2. Train with recommended model (Omnilingual CTC-300M)
python train.py \
    --model_family omnilingual \
    --cv_base_dir ./data/cv-corpus-24.0-2025-12-05/tw \
    --num_epochs 1 \
    --batch_size 8 \
    --grad_accum 4 \
    --use_lora \
    --train_kenlm \
    --pseudo_label \
    --bf16

# 3. Serve inference
python inference_server.py --model_path ./reachy-akan-asr/final
```

---

## Why This Pipeline?

Your original notebook had **critical bugs** that silently destroyed WER:

| Issue | Impact | Status |
|-------|--------|--------|
| Language token = "yoruba" (not Twi) | +20-40% WER | **Fixed** |
| Model size inconsistency (small vs medium) | Reproducibility failure | **Fixed** |
| Farmerline double-counted | Biased training data | **Fixed** |
| Only 1000 steps (68% data unseen) | Severe under-training | **Fixed** |
| Test set used for eval + model selection | Data leakage | **Fixed** |
| No text normalization for WER | Uncomparable metrics | **Fixed** |
| No CER (can't track diacritics) | Blind to ɛ/ɔ/ŋ errors | **Fixed** |
| Streaming datasets (no caching) | 5-10x slower training | **Fixed** |
| Apostrophe stripped from Twi text | Destroys elision markers | **Fixed** |

---

## Model Comparison for Akan/Twi ASR

Based on [Azunre et al. (2025)](https://arxiv.org/abs/2512.10968) benchmarking 13 African languages:

### Benchmark Results on Akan (Ashesi Financial Dataset)

| Model | Params | WER @ 1h | WER @ 5h | WER @ 50h | WER @ 100h | Notes |
|-------|--------|----------|----------|-----------|------------|-------|
| **MMS-1B** | 1B | 98.8% | 37.4% | ~32% | **31.1%** | Has Akan adapter! |
| **Whisper-small** | 244M | ~50% | ~35% | ~32% | ~30% | NO Akan tokenizer |
| **XLS-R 300M** | 300M | 45.6% | ~35% | 30.0% | ~30% | Best scaling |
| **W2v-BERT** | 300M | ~48% | ~37% | ~34% | ~32% | Good but plateaus |
| **Omnilingual CTC-300M** | 325M | *NEW* | *NEW* | *NEW* | **Target: <20%** | **1,600 langs, Apache 2.0** |

### Key Insights from the Literature

1. **All models plateau at ~30% WER** for Akan with 100h of labeled data
2. **To break 10% WER**, you MUST:
   - Pseudo-label the **109k unlabeled WaxalNLP clips** (~300h additional data)
   - Use **KenLM 4-gram fusion** (proven 5-20% relative improvement)
   - Apply **SpecAugment + speed perturbation**
   - Train for **4,000-5,000+ steps** (not 1,000!)
3. **Omnilingual ASR** (Nov 2025) supports 1,600 languages explicitly - most promising
4. **MMS** has built-in Akan adapter support - only trains adapter weights (very efficient)

### Our Recommendation: Start with Omnilingual CTC-300M

| Criterion | Omnilingual CTC-300M | MMS-1B | XLS-R 300M |
|-----------|---------------------|--------|------------|
| **Akan support** | Explicit (aka_Latn) | Explicit (aka adapter) | Generic multilingual |
| **Parameters** | 325M | 1B | 300M |
| **Training VRAM** | ~4GB | ~8GB | ~4GB |
| **Inference VRAM** | ~2GB | ~3GB | ~2GB |
| **Inference speed** | 96x real-time | 16x real-time | 48x real-time |
| **License** | Apache 2.0 | CC-BY-NC 4.0 | Apache 2.0 |
| **Fine-tuning** | Full or LoRA | Adapter only | Full or LoRA |

**Why not Whisper as the primary path?** Whisper does NOT have an Akan/Twi language token in its built-in language vocabulary. For low-resource local-language ASR, I would not anchor the project on a neighboring language token hack. Omnilingual and MMS are better starting points because they explicitly support Akan and keep the training/evaluation story cleaner.

---

## Architecture

```
reachy_asr/
|-- config.py              # Centralized configuration (Data, Model, Training, Aug, LM, Eval)
|-- data_pipeline.py       # Dataset loading + preprocessing + disk caching
|-- models.py              # Multi-backend model factory (Omnilingual, MMS, XLS-R, W2v-BERT)
|-- augmentation.py        # SpecAugment, speed perturbation, noise injection
|-- evaluation.py          # Proper WER/CER with Twi text normalization
|-- lm_fusion.py           # KenLM training + pyctcdecode CTC fusion
|-- train.py               # Main training script with progressive training
|-- inference_server.py    # FastAPI production inference server
|-- Dockerfile             # Multi-stage container for GPU cloud
|-- modal_deploy.py        # Modal.com deployment (training + inference)
|-- runpod_handler.py      # RunPod serverless handler
```

### Data Flow

```
Raw Datasets                    Preprocessed Cache              Training
------------                    ----------------                --------
WaxalNLP (aka_asr)    ──┐
                          ├──→  Disk cache  ──→  .map()  ──→  GPU
Farmerline 2.0        ──┤      (parquet)         (features)
                          │                          │
Common Voice (MDC)    ──┘                          ↓
                                              SpecAugment
WaxalNLP (unlabeled)  ──→  Pseudo-labeling  ──→  + train set
                              (model-generated)
                                                    ↓
Training text corpus  ──→  KenLM 4-gram  ──→  CTC decoding
```

---

## Installation

### Requirements

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- 8GB+ GPU VRAM (for 300M models)
- 40GB+ disk space for datasets

### Setup

```bash
# Clone
git clone <your-repo>
cd reachy-asr

# Create environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Optional: KenLM for LM fusion
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install pyctcdecode

# Optional: Omnilingual ASR
pip install omnilingual-asr
```

### Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key  # optional
export WANDB_PROJECT=reachy-akan-asr
```

### RunPod Pod Environment Variables

If you train on a **RunPod GPU Pod** and want the variables available in the pod terminal, notebooks, and Python processes, set them in the **Pod configuration / Template environment variables**, not only inside a notebook cell.

Recommended variables:

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=reachy-akan-asr
HF_HOME=/workspace/.cache/huggingface
HF_DATASETS_CACHE=/workspace/.cache/datasets
TRANSFORMERS_CACHE=/workspace/.cache/transformers
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOKENIZERS_PARALLELISM=false
MODEL_OUTPUT_DIR=/workspace/reachy-akan-asr
CV_BASE_DIR=/workspace/data/cv-corpus-24.0-2025-12-05/tw
```

What to know on RunPod:

- If you edit Pod environment variables later, the Pod restarts. Anything outside `/workspace` can be lost.
- A running Jupyter kernel may not see newly-added variables until you restart the kernel.
- To verify from the terminal, run `echo $HF_TOKEN`, `echo $CV_BASE_DIR`, or `env | grep HF_`.
- To verify from a notebook cell:

```python
import os
print(os.environ.get("HF_TOKEN"))
print(os.environ.get("CV_BASE_DIR"))
```

If the terminal sees the variable but the notebook does not, restart the notebook kernel. If neither sees it, stop and recreate or restart the Pod after saving the environment variables in RunPod.

---

## Data Preparation

### 1. WaxalNLP + Farmerline (auto-downloaded from HuggingFace)

These are downloaded automatically on first run and cached to disk.

### 2. Common Voice Twi (manual download from Mozilla Data Collective)

Since October 2025, Common Voice is **exclusively** available through [Mozilla Data Collective](https://datacollective.mozillafoundation.org/):

```bash
# 1. Create account at https://datacollective.mozillafoundation.org/
# 2. Generate API key
# 3. Download Twi dataset via API or browser
# 4. Extract to: ./data/cv-corpus-24.0-2025-12-05/tw/

# Expected structure:
# ./data/cv-corpus-24.0-2025-12-05/tw/
#   ├── clips/
#   │   ├── common_voice_tw_00000001.mp3
#   │   └── ...
#   ├── train.tsv
#   ├── test.tsv
#   └── validated.tsv
```

---

## Training

### First Run: Smoke Test on RunPod

Do this before a full 10-15 epoch run. It confirms that dataset loading, Hub auth, output paths, and VRAM are all correct.

```bash
python train.py \
    --model_family omnilingual \
    --cv_base_dir ${CV_BASE_DIR:-./data/cv-corpus-24.0-2025-12-05/tw} \
    --output_dir ${MODEL_OUTPUT_DIR:-./reachy-akan-asr} \
    --num_epochs 1 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 6.25e-6 \
    --use_lora \
    --pseudo_label \
    --train_kenlm \
    --bf16 \
    --push_to_hub \
    --hub_model_id teckedd/reachy-akan-asr
```

### Recommended Full Pipeline (Omnilingual + LoRA + Pseudo-Labeling + KenLM)

```bash
python train.py \
    --model_family omnilingual \
    --cv_base_dir ${CV_BASE_DIR:-./data/cv-corpus-24.0-2025-12-05/tw} \
    --output_dir ${MODEL_OUTPUT_DIR:-./reachy-akan-asr} \
    --num_epochs 15 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 6.25e-6 \
    --scheduler cosine \
    --warmup_ratio 0.1 \
    --use_lora \
    --lora_rank 64 \
    --progressive \
    --spec_augment \
    --pseudo_label \
    --train_kenlm \
    --bf16 \
    --wandb_project reachy-akan-asr \
    --push_to_hub \
    --hub_model_id teckedd/reachy-akan-asr
```

Notes:

- `batch_size=8` with `grad_accum=4` is the safer starting point on a 24 GB RTX 4090.
- The preprocessing cache schema was updated. Your next run will rebuild preprocessed datasets once.
- If you want a cheaper first validation run, keep the same config and use `--num_epochs 1`.

### Progressive Training (Two-Phase)

Phase 1: Freeze encoder, train decoder/head only (~5 epochs, lr=6.25e-6)
Phase 2: Unfreeze all, full fine-tuning (~5 epochs, lr=3.125e-6)

This is enabled by `--progressive` flag.

### MMS with Akan Adapter

```bash
python train.py \
    --model_family mms \
    --use_adapter \
    --cv_base_dir ./data/cv-corpus-24.0-2025-12-05/tw \
    --num_epochs 10 \
    --batch_size 8 \
    --lr 5e-5
```

MMS adapter training updates **only ~0.1% of parameters** - extremely efficient.

### What I Would Do for Local-Language ASR

If I were training ASR for local languages like Twi, Ga, Ewe, Dagbani, Fante, or code-switched Ghanaian speech, I would do this:

1. Start with a model that explicitly supports the language or language family.
   Omnilingual CTC first, MMS second, XLS-R third.

2. Build one clean supervised corpus before chasing scale.
   Fix text normalization, remove duplicate clips, unify sample rates, keep apostrophes/diacritics, and hold out a real test set by speaker and domain.

3. Fine-tune with LoRA on a 24 GB GPU first.
   This is the cheapest way to validate the full pipeline.

4. Add unlabeled audio only after the supervised baseline is stable.
   Pseudo-labeling is useful, but only after you trust the base model and evaluation loop.

5. Train a small external language model from normalized transcripts.
   For low-resource ASR, decoding quality often improves materially with a clean n-gram LM.

6. Evaluate by domain, not only overall WER.
   Healthcare speech, call-center speech, radio speech, and conversational audio behave very differently.

7. Only revisit Whisper if you have a clear reason.
   Whisper is still useful as a baseline or for transfer, but I would not make it the primary Akan/Twi path because the language-token story is wrong for this task.

### Resume from Checkpoint

```bash
python train.py \
    --resume ./reachy-akan-asr/checkpoint-5000 \
    --num_epochs 10
```

### RunPod Workflow

```bash
# 1. Clone your repo onto the pod
git clone <your-repo-url> /workspace/reachy_asr
cd /workspace/reachy_asr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm your environment variables are visible
echo $HF_TOKEN
echo $CV_BASE_DIR
python - <<'PY'
import os
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
print("CV_BASE_DIR:", os.environ.get("CV_BASE_DIR"))
PY

# 4. Run smoke test
python train.py \
  --model_family omnilingual \
  --cv_base_dir ${CV_BASE_DIR} \
  --output_dir ${MODEL_OUTPUT_DIR:-/workspace/reachy-akan-asr} \
  --num_epochs 1 \
  --batch_size 8 \
  --grad_accum 4 \
  --use_lora \
  --pseudo_label \
  --train_kenlm \
  --bf16 \
  --push_to_hub \
  --hub_model_id teckedd/reachy-akan-asr
```

---

## Expected Results

### With ~33h labeled data only

| Model | Val WER | Test WER | Training Time (A100) |
|-------|---------|----------|---------------------|
| Omnilingual CTC-300M | ~25-30% | ~28-35% | ~2 hours |
| MMS-1B (adapter) | ~28-32% | ~30-35% | ~3 hours |
| XLS-R 300M | ~28-33% | ~30-35% | ~2 hours |

### With pseudo-labeling (~300h additional)

| Model | Val WER | Test WER | Training Time (A100) |
|-------|---------|----------|---------------------|
| Omnilingual CTC-300M | ~15-20% | ~18-25% | ~8 hours |
| MMS-1B (adapter) | ~18-22% | ~20-28% | ~10 hours |
| XLS-R 300M | ~18-22% | ~20-28% | ~8 hours |

### With pseudo-labeling + KenLM fusion

| Model | Val WER | Test WER | Training Time (A100) |
|-------|---------|----------|---------------------|
| Omnilingual CTC-300M | **~12-16%** | **~15-20%** | ~8 hours + 30min LM |
| MMS-1B (adapter) | **~15-18%** | **~18-22%** | ~10 hours + 30min LM |
| XLS-R 300M | **~15-18%** | **~18-22%** | ~8 hours + 30min LM |

> **Note:** These are estimates. Actual WER depends heavily on data quality, domain match, and hyperparameter tuning. Healthcare domain speech (symptoms, complaints) may differ from general Twi speech.

---

## Deployment

### Docker (Recommended)

```bash
# Build
docker build -t reachy-asr .

# Training
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    -e HF_TOKEN=$HF_TOKEN \
    reachy-asr \
    python train.py --model_family omnilingual --cv_base_dir /app/data/cv-corpus-24.0-2025-12-05/tw

# Inference
docker run --gpus all -p 8000:8000 \
    -e MODEL_PATH=/app/output/reachy-akan-asr/final \
    reachy-asr \
    python inference_server.py --host 0.0.0.0 --port 8000
```

### Modal.com

```bash
# Install Modal
pip install modal
modal login

# Set secrets
modal secret create hf-token HF_TOKEN=your_token
modal secret create wandb-key WANDB_API_KEY=your_key

# Run training
modal run modal_deploy.py \
    --model-family omnilingual \
    --cv-base-dir /data/cv \
    --num-epochs 15

# Deploy inference API
modal deploy modal_deploy.py
```

### RunPod

```bash
# 1. Upload Docker image to RunPod registry
# 2. Create serverless endpoint with:
#    - GPU: RTX 4090 or A100
#    - Container: your-image
#    - Handler: /app/runpod_handler.py
# 3. Set environment variables in endpoint config
```

### Direct Cloud (RunPod/Any GPU Instance)

```bash
# SSH into GPU instance
git clone <repo>
cd reachy-asr
pip install -r requirements.txt

# Run training (nohup for persistence)
nohup python train.py \
    --model_family omnilingual \
    --cv_base_dir /workspace/data/cv \
    --use_lora \
    --pseudo_label \
    --train_kenlm \
    > training.log 2>&1 &

# Monitor
tail -f training.log
watch -n 5 nvidia-smi
```

---

## Inference API

### Start Server

```bash
python inference_server.py --model_path ./reachy-akan-asr/final --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/transcribe` | POST | Transcribe audio file |

### Example Request

```bash
curl -X POST \
    -F "audio_file=@patient_symptoms.wav" \
    http://localhost:8000/transcribe

# Response:
# {
#   "text": "me yare me ti me hu akoma mu yaw",
#   "duration_sec": 5.2,
#   "sample_rate": 16000,
#   "language": "tw"
# }
```

### Python Client

```python
import requests

def transcribe(audio_path: str, api_url: str = "http://localhost:8000"):
    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{api_url}/transcribe",
            files={"audio_file": f}
        )
    return response.json()["text"]

text = transcribe("patient_symptoms.wav")
print(text)  # "me yare me ti me hu akoma mu yaw"
```

---

## Twi Text Normalization

The pipeline preserves **linguistically significant** Twi characters:

| Character | Role | Handled |
|-----------|------|---------|
| `ɛ` / `Ɛ` | Open 'e' (as in 'bed') | Preserved |
| `ɔ` / `Ɔ` | Open 'o' (as in 'law') | Preserved |
| `ŋ` / `Ŋ` | Velar nasal (ng sound) | Preserved |
| `'` | Elision marker (`w'ani`) | Preserved |
| `-` | Compound word separator | Preserved |
| `. , ? ! ; :` | Punctuation | Removed for WER |

Example:
```
Original:   "Gɛls fɔr egyina ha obiaa kita buk ɔmo gyina pila kɛseɛ be ho baako abɔ ne tii tententen ."
Normalized: "gɛls fɔr egyina ha obiaa kita buk ɔmo gyina pila kɛseɛ be ho baako abɔ ne tii tententen"
```

---

## Caching Strategy

Every expensive operation is cached to disk:

```
cache/
├── huggingface/           # HF model/tokenizer cache
├── datasets/
│   ├── waxal_train_raw/   # Downloaded WaxalNLP
│   ├── waxal_test_raw/
│   ├── farmerline_train_raw/
│   ├── farmerline_test_raw/
│   ├── cv_train_raw/      # Downloaded Common Voice
│   └── cv_test_raw/
└── transformers/          # Model checkpoints

preprocessed/
├── train_omnilingual_preprocessed_{hash}/  # Preprocessed features
├── validation_omnilingual_preprocessed_{hash}/
├── test_omnilingual_preprocessed_{hash}/
└── fulldataset_omnilingual_{hash}/         # Combined dataset
```

Cache is automatically invalidated when config changes (hash-based).

---

## Cost Estimates (RunPod GPU Cloud)

| Configuration | GPU | VRAM | Cost/hr | Total Cost |
|--------------|-----|------|---------|------------|
| Budget training | RTX 4090 | 24GB | $0.44 | ~$4-8 |
| Standard training | A100 40GB | 40GB | $1.19 | ~$10-24 |
| Fast training | H100 | 80GB | $2.49 | ~$20-50 |
| Inference | A10G | 24GB | $0.50 | ~$10/month always-on |
| Serverless inference | T4 | 16GB | $0.20 | ~$0.001/request |

---

## Roadmap to WER < 10%

```
Phase 1: Fix the foundation (Week 1)
  - Fix language token, splits, normalization
  - Increase steps to 5000+
  - Expected WER: 25-35% → 15-25%

Phase 2: Scale the data (Week 2-3)
  - Pseudo-label 109k unlabeled clips
  - SpecAugment + speed perturbation
  - Expected WER: 15-25% → 12-20%

Phase 3: LM fusion + tuning (Week 3-4)
  - Train KenLM 4-gram
  - Tune beam search + alpha/beta
  - Expected WER: 12-20% → 8-15%

Phase 4: Domain adaptation (Week 4-6)
  - Collect healthcare-specific Twi audio
  - Fine-tune on medical vocabulary
  - Expected WER: 8-15% → 5-10%
```

---

## References

1. **Azunre et al. (2025)** - [Benchmarking ASR for African Languages](https://arxiv.org/abs/2512.10968)
2. **Meta Omnilingual ASR (2025)** - [1,600 languages](https://github.com/facebookresearch/omnilingual-asr)
3. **Meta MMS (2024)** - [Massively Multilingual Speech](https://huggingface.co/facebook/mms-1b-all)
4. **Whisper (2023)** - [Robust Speech Recognition](https://github.com/openai/whisper)
5. **XLS-R (2021)** - [Cross-lingual Speech Representations](https://huggingface.co/facebook/wav2vec2-xls-r-300m)

---

## License

This pipeline: MIT License

Model licenses:
- Omnilingual ASR: Apache 2.0
- MMS: CC-BY-NC 4.0 (non-commercial)
- XLS-R: Apache 2.0
- Whisper: MIT
