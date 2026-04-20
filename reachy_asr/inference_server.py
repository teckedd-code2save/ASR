#!/usr/bin/env python3
"""
ReachyAI ASR Inference Server
==============================
FastAPI-based transcription server for production deployment.
Supports multiple model backends with GPU acceleration.

Usage:
    # Start server
    python inference_server.py --model_path ./reachy-akan-asr/final

    # Test
    curl -X POST -F "audio=@sample.wav" http://localhost:8000/transcribe
"""

import os
import base64
import tempfile
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import ModelConfig
from models import ASRModelFactory
from data_pipeline import normalize_text


# ====================================================================
# GLOBALS (loaded at startup)
# ====================================================================

model = None
processor = None
device = None
model_family = None


class Base64AudioRequest(BaseModel):
    audio_base64: str


def load_model_for_inference(model_path: str, device_str: str = "cuda"):
    """Load model and processor for inference."""
    global model, processor, device, model_family
    
    device = device_str if torch.cuda.is_available() else "cpu"
    
    # Detect model family from config or path
    config = ModelConfig()
    factory = ASRModelFactory(config)
    
    # Try loading as saved checkpoint first
    if Path(model_path).exists():
        from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = AutoModelForCTC.from_pretrained(model_path, trust_remote_code=True)
            model_family = "ctc"
        except Exception:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            model_family = "seq2seq"
        model.to(device)
    else:
        # Load from HuggingFace hub
        model, processor, model_family = factory.load_model(
            model_family=config.model_family,
            device=device
        )
    
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"Device: {device}")
    return model, processor


# ====================================================================
# AUDIO PREPROCESSING
# ====================================================================

def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000):
    """
    Load and preprocess audio from bytes.
    
    Supports: WAV, MP3, OGG, FLAC, M4A
    """
    import soundfile as sf
    import librosa
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Load audio
        audio, sr = sf.read(tmp_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz
        if sr != target_sr:
            audio = librosa.resample(
                audio.astype(np.float64),
                orig_sr=sr,
                target_sr=target_sr
            ).astype(np.float32)
        
        return audio, target_sr
    
    finally:
        os.unlink(tmp_path)


# ====================================================================
# TRANSCRIPTION
# ====================================================================

def transcribe_audio(audio_array: np.ndarray) -> str:
    """
    Transcribe audio array to text.
    
    Args:
        audio_array: Audio at 16kHz
    
    Returns:
        Transcription text
    """
    import torch
    
    with torch.no_grad():
        # Process audio
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        if model_family == "ctc":
            # CTC model: logits -> argmax -> decode
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(pred_ids)[0]
        else:
            # Seq2seq: generate
            pred_ids = model.generate(**inputs, max_length=225)
            text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    
    # Normalize output
    text = normalize_text(text)
    
    return text


# ====================================================================
# FASTAPI APP
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model."""
    model_path = os.environ.get("MODEL_PATH", "./reachy-akan-asr/final")
    load_model_for_inference(model_path)
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="ReachyAI Akan ASR",
    description="Twi/Akan speech-to-text API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_family": model_family,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/transcribe")
async def transcribe_endpoint(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file.
    
    - **audio_file**: Audio file (WAV, MP3, OGG, FLAC)
    
    Returns JSON with transcription text.
    """
    if model is None:
        raise HTTPException(500, "Model not loaded")
    
    # Validate file type
    allowed_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg",
                     "audio/flac", "audio/x-flac", "audio/mp3", "audio/m4a",
                     "application/octet-stream"}
    
    if audio_file.content_type not in allowed_types:
        raise HTTPException(400, f"Unsupported audio format: {audio_file.content_type}")
    
    try:
        # Read audio
        audio_bytes = await audio_file.read()
        
        # Preprocess
        audio_array, sr = preprocess_audio(audio_bytes)
        
        # Check duration
        duration = len(audio_array) / sr
        if duration > 60:
            raise HTTPException(400, f"Audio too long: {duration:.1f}s (max 60s)")
        
        # Transcribe
        text = transcribe_audio(audio_array)
        
        return {
            "text": text,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "language": "tw"  # Twi/Akan
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.post("/transcribe-base64")
async def transcribe_base64_endpoint(request: Base64AudioRequest):
    """Transcribe base64-encoded audio for backend/mobile clients."""
    if model is None:
        raise HTTPException(500, "Model not loaded")

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array, sr = preprocess_audio(audio_bytes)

        duration = len(audio_array) / sr
        if duration > 60:
            raise HTTPException(400, f"Audio too long: {duration:.1f}s (max 60s)")

        text = transcribe_audio(audio_array)
        return {
            "text": text,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "language": "tw"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "ReachyAI Akan ASR Server", "docs": "/docs"}


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./reachy-akan-asr/final")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    
    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )


if __name__ == "__main__":
    main()
