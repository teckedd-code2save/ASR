#!/usr/bin/env python3
"""
ReachyAI ASR RunPod Handler
============================
RunPod serverless/endpoint handler for Akan/Twi ASR.
Optimized for RunPod's serverless GPU infrastructure.

Usage:
    # Deploy as RunPod serverless endpoint
    # Set environment variables:
    #   MODEL_PATH=/app/output/reachy-akan-asr/final
    #   HF_TOKEN=your_token
"""

import os
import sys
import base64
import tempfile

import numpy as np
import torch
import runpod  # pip install runpod

# Add app to path
sys.path.insert(0, "/app")

from config import ModelConfig
from models import ASRModelFactory
from data_pipeline import normalize_text


# ====================================================================
# MODEL LOADING (once at cold start)
# ====================================================================

class ASRHandler:
    """RunPod handler with lazy model loading."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.model_family = None
        self._loaded = False
    
    def load(self):
        """Load model if not already loaded."""
        if self._loaded:
            return
        
        model_path = os.environ.get("MODEL_PATH", "/app/output/reachy-akan-asr/final")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        
        # Load processor
        from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        try:
            self.model = AutoModelForCTC.from_pretrained(model_path, trust_remote_code=True)
            self.model_family = "ctc"
        except Exception:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model_family = "seq2seq"
        
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        
        print("Model loaded successfully")
    
    def transcribe(self, audio_array: np.ndarray) -> str:
        """Transcribe audio to text."""
        self.load()
        
        with torch.no_grad():
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.model_family == "ctc":
                logits = self.model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                text = self.processor.batch_decode(pred_ids)[0]
            else:
                pred_ids = self.model.generate(**inputs, max_length=225)
                text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        
        return normalize_text(text)


# Global handler instance (persisted across requests)
handler = ASRHandler()


# ====================================================================
# RUNPOD HANDLER
# ====================================================================

def process_request(job):
    """
    RunPod serverless handler.
    
    Expected input format:
    {
        "audio_base64": "<base64-encoded-audio>",
        "audio_format": "wav"  # optional: wav, mp3, ogg
    }
    
    Or:
    {
        "audio_url": "https://..."  # URL to audio file
    }
    """
    job_input = job.get("input", {})
    
    try:
        import librosa
        import soundfile as sf
        
        audio_data = None
        sr = 16000
        
        # Handle base64 audio
        if "audio_base64" in job_input:
            audio_bytes = base64.b64decode(job_input["audio_base64"])
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                audio_data, sr = librosa.load(tmp_path, sr=16000)
            finally:
                os.unlink(tmp_path)
        
        # Handle audio URL
        elif "audio_url" in job_input:
            import requests
            response = requests.get(job_input["audio_url"], timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                audio_data, sr = librosa.load(tmp_path, sr=16000)
            finally:
                os.unlink(tmp_path)
        
        else:
            return {"error": "No audio provided. Use 'audio_base64' or 'audio_url'"}
        
        # Check duration
        duration = len(audio_data) / sr
        if duration > 60:
            return {"error": f"Audio too long: {duration:.1f}s (max 60s)"}
        
        # Transcribe
        text = handler.transcribe(audio_data)
        
        return {
            "transcription": text,
            "duration_sec": round(duration, 2),
            "language": "tw",
            "language_name": "Twi/Akan"
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    runpod.serverless({"handler": process_request})
