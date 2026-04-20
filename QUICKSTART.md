# Quickstart Deployment Guide

## Recommended Path

Train on a **RunPod GPU Pod** with an **RTX 4090 24 GB** or **RTX 3090 24 GB**.

Use this repo's `reachy_asr/train.py` pipeline with:

- `--model_family omnilingual`
- `--use_lora`
- `--pseudo_label`
- `--train_kenlm`

Do **not** follow the older Whisper-only flow for this repo. The current training path is centered on Omnilingual and MMS because they explicitly support Akan.

## RunPod Setup

### Step 1: Create the Pod

In RunPod:

1. Go to `Deploy -> GPU Pods`
2. Pick `Community Cloud` if you want the lowest cost
3. Choose one of:
   - `RTX 4090 24 GB`
   - `RTX 3090 24 GB`
4. Use an official PyTorch/Jupyter-style template
5. Set:
   - Container disk: `30 GB+`
   - Volume/workspace disk: `50 GB+` if available

### Step 2: Set Pod Environment Variables

Set these in the **Pod configuration** or **template** before launching:

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

Important:

- If you edit env vars later, RunPod restarts the Pod.
- If your notebook kernel was already running, restart the kernel before expecting new env vars there.
- Keep data and outputs under `/workspace`.

### Step 3: Verify Environment Variables

From a pod terminal:

```bash
echo $HF_TOKEN
echo $CV_BASE_DIR
env | grep HF_
```

From a notebook cell:

```python
import os
print(os.environ.get("HF_TOKEN"))
print(os.environ.get("CV_BASE_DIR"))
```

If the terminal sees the variables but the notebook does not, restart the kernel.

## Install and Prepare

```bash
cd /workspace
git clone <your-repo-url> reachy_asr
cd /workspace/reachy_asr
pip install -r requirements.txt
```

Download Common Voice Twi and extract it to:

```bash
/workspace/data/cv-corpus-24.0-2025-12-05/tw
```

Expected structure:

```text
/workspace/data/cv-corpus-24.0-2025-12-05/tw/
├── clips/
├── train.tsv
├── test.tsv
└── validated.tsv
```

## Smoke Test

Run this first:

```bash
python reachy_asr/train.py \
  --model_family omnilingual \
  --cv_base_dir ${CV_BASE_DIR} \
  --output_dir ${MODEL_OUTPUT_DIR:-/workspace/reachy-akan-asr} \
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

This confirms:

- dataset loading works
- Common Voice path is correct
- HF auth is available
- output path is writable
- the GPU can handle the configuration

## Full Training Run

If the smoke test works:

```bash
python reachy_asr/train.py \
  --model_family omnilingual \
  --cv_base_dir ${CV_BASE_DIR} \
  --output_dir ${MODEL_OUTPUT_DIR:-/workspace/reachy-akan-asr} \
  --num_epochs 10 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 6.25e-6 \
  --scheduler cosine \
  --warmup_ratio 0.1 \
  --use_lora \
  --lora_rank 64 \
  --progressive \
  --pseudo_label \
  --train_kenlm \
  --bf16 \
  --wandb_project reachy-akan-asr \
  --push_to_hub \
  --hub_model_id teckedd/reachy-akan-asr
```

Use `--num_epochs 15` only after you confirm the 10-epoch run is stable and worth extending.

## Monitoring

```bash
watch -n 5 nvidia-smi
```

If you want logs in a file:

```bash
python reachy_asr/train.py ... 2>&1 | tee training.log
```

## Serve for Reachy Health

After training:

```bash
python reachy_asr/inference_server.py \
  --model_path /workspace/reachy-akan-asr/final \
  --host 0.0.0.0 \
  --port 8000
```

Available endpoints:

- `GET /health`
- `POST /transcribe`
- `POST /transcribe-base64`

## Notes

- Omnilingual is the recommended first path for Akan/Twi.
- MMS is the backup path if you want adapter-based fine-tuning.
- Whisper can still be used as a baseline, but it is not the main path for this repo.
