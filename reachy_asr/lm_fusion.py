"""
ReachyAI ASR Language Model Fusion
====================================
KenLM n-gram language model training and CTC decoding fusion.
Proven 5-20% relative WER improvement for low-resource ASR.

References:
- KenLM: https://github.com/kpu/kenlm
- pyctcdecode: https://github.com/kensho-technologies/pyctcdecode
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from transformers import Wav2Vec2Processor

from config import LMConfig, DataConfig


# ====================================================================
# KENLM INSTALLATION CHECK
# ====================================================================

def check_kenlm_installed() -> bool:
    """Check if KenLM is installed and available."""
    try:
        import kenlm
        return True
    except ImportError:
        return False


def install_kenlm_instructions():
    """Print installation instructions for KenLM."""
    print("""
KenLM is not installed. Install it:

1. Build from source:
   git clone https://github.com/kpu/kenlm.git
   cd kenlm
   mkdir build && cd build
   cmake ..
   make -j$(nproc)

2. Install Python bindings:
   pip install https://github.com/kpu/kenlm/archive/master.zip

3. Or install pyctcdecode (includes KenLM):
   pip install pyctcdecode
""")


# ====================================================================
# LANGUAGE MODEL TRAINING
# ====================================================================

class LanguageModelTrainer:
    """Train KenLM n-gram language model from text corpus."""
    
    def __init__(self, config: Optional[LMConfig] = None):
        self.config = config or LMConfig()
        self.has_kenlm = check_kenlm_installed()
    
    def prepare_text_corpus(
        self,
        transcriptions: List[str],
        output_path: str,
        lowercase: bool = True,
        normalize_unicode: bool = True
    ) -> str:
        """
        Prepare text corpus for KenLM training.
        
        Args:
            transcriptions: List of text strings
            output_path: Where to write the corpus file
            lowercase: Convert to lowercase
            normalize_unicode: Apply Unicode NFC normalization
        
        Returns:
            Path to the prepared corpus file
        """
        import unicodedata
        
        corpus_path = Path(output_path)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for text in transcriptions:
                if not text or not text.strip():
                    continue
                
                # Normalize
                if normalize_unicode:
                    text = unicodedata.normalize('NFC', text)
                
                # Lowercase
                if lowercase:
                    text = text.lower()
                
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                if text:
                    f.write(text + '\n')
        
        # Report statistics
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = sum(len(line.split()) for line in lines)
        
        print(f"Corpus prepared: {len(lines)} lines, {words} words")
        print(f"  Saved to: {corpus_path}")
        
        return str(corpus_path)
    
    def train_kenlm(
        self,
        corpus_path: str,
        output_binary_path: str,
        order: Optional[int] = None
    ) -> str:
        """
        Train KenLM n-gram language model.
        
        Args:
            corpus_path: Path to text corpus
            output_binary_path: Path for output binary LM
            order: N-gram order (default from config)
        
        Returns:
            Path to trained binary LM
        """
        order = order or self.config.kenlm_order
        
        # Check for lmplz binary
        lmplz = self.config.kenlm_binary
        if not Path(lmplz).exists():
            # Try system path
            lmplz = "lmplz"
        
        build_binary = self.config.kenlm_build_binary
        if not Path(build_binary).exists():
            build_binary = "build_binary"
        
        # Check tools exist
        try:
            subprocess.run([lmplz, "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"ERROR: lmplz not found at {lmplz}")
            print("Install KenLM: https://github.com/kpu/kenlm")
            return None
        
        output_path = Path(output_binary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            arpa_path = f"{tmpdir}/model.arpa"
            
            print(f"\nTraining {order}-gram KenLM...")
            print(f"  Corpus: {corpus_path}")
            print(f"  Output: {output_binary_path}")
            
            # Step 1: Train ARPA format
            cmd = [
                lmplz,
                "-o", str(order),
                "--arpa", arpa_path,
                "--discount_fallback",  # For small corpora
                "--memory", "2G"
            ]
            
            with open(corpus_path, 'r', encoding='utf-8') as f_in:
                result = subprocess.run(
                    cmd,
                    stdin=f_in,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                print(f"KenLM training failed:\n{result.stderr}")
                return None
            
            # Step 2: Convert to binary format (faster loading)
            binary_path = str(output_path)
            cmd = [build_binary, "-s", arpa_path, binary_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Binary conversion failed, keeping ARPA: {result.stderr}")
                binary_path = arpa_path
            
            print(f"  LM trained: {binary_path}")
            print(f"  Size: {os.path.getsize(binary_path) / 1e6:.1f} MB")
            
            return binary_path
    
    def build_vocabulary(self, transcriptions: List[str]) -> List[str]:
        """
        Build vocabulary from training transcriptions.
        Needed for CTC decoding with pyctcdecode.
        
        Returns:
            List of unique words sorted by frequency
        """
        word_counts = {}
        for text in transcriptions:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        vocab = [w for w, _ in sorted(word_counts.items(), key=lambda x: -x[1])]
        return vocab


# ====================================================================
# CTC DECODING WITH LM FUSION
# ====================================================================

class LMEnhancedDecoder:
    """
    CTC decoder with language model fusion using pyctcdecode.
    
    Provides:
    - Greedy decoding (baseline)
    - Beam search decoding
    - KenLM-enhanced beam search decoding
    """
    
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        kenlm_path: Optional[str] = None,
        vocab: Optional[List[str]] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        beam_width: int = 100
    ):
        self.processor = processor
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        
        # Try to use pyctcdecode
        try:
            from pyctcdecode import build_ctcdecoder
            self.pyctcdecode_available = True
        except ImportError:
            print("WARNING: pyctcdecode not installed. LM fusion disabled.")
            print("  pip install pyctcdecode")
            self.pyctcdecode_available = False
            return
        
        # Build vocabulary from tokenizer
        if vocab is None:
            vocab = self._build_vocab_from_tokenizer()
        
        # Build decoder
        self._build_decoder(kenlm_path, vocab)
    
    def _build_vocab_from_tokenizer(self) -> List[str]:
        """Build vocabulary list from tokenizer."""
        vocab_dict = self.processor.tokenizer.get_vocab()
        
        # Convert token IDs to characters/words
        id_to_token = {v: k for k, v in vocab_dict.items()}
        
        # Sort by token ID
        vocab = [id_to_token[i] for i in range(len(id_to_token))]
        
        return vocab
    
    def _build_decoder(self, kenlm_path: Optional[str], vocab: List[str]):
        """Build pyctcdecode decoder."""
        from pyctcdecode import build_ctcdecoder
        
        # Load KenLM if available
        kenlm_model = None
        if kenlm_path and Path(kenlm_path).exists():
            try:
                kenlm_model = kenlm_path
                print(f"  Loaded KenLM: {kenlm_path}")
            except Exception as e:
                print(f"  Failed to load KenLM: {e}")
        
        # Build decoder
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=kenlm_model,
            alpha=self.alpha,
            beta=self.beta,
            beam_width=self.beam_width
        )
    
    def decode(
        self,
        logits: np.ndarray,
        use_lm: bool = True
    ) -> str:
        """
        Decode CTC logits to text.
        
        Args:
            logits: Model output logits (time, vocab_size)
            use_lm: Whether to use LM-enhanced decoding
        
        Returns:
            Decoded text string
        """
        if not self.pyctcdecode_available or not use_lm:
            # Fallback: greedy decoding
            pred_ids = np.argmax(logits, axis=-1)
            return self.processor.decode(pred_ids)
        
        # pyctcdecode beam search
        text = self.decoder.decode(
            logits,
            beam_width=self.beam_width if use_lm else 1
        )
        
        return text
    
    def decode_batch(
        self,
        logits_batch: np.ndarray,
        use_lm: bool = True
    ) -> List[str]:
        """
        Decode batch of CTC logits.
        
        Args:
            logits_batch: (batch, time, vocab_size)
            use_lm: Whether to use LM-enhanced decoding
        
        Returns:
            List of decoded text strings
        """
        return [self.decode(l, use_lm=use_lm) for l in logits_batch]


# ====================================================================
# FULL PIPELINE: TRAIN LM + DECODE
# ====================================================================

def setup_lm_fusion(
    processor: Wav2Vec2Processor,
    train_transcriptions: List[str],
    output_dir: str = "./lm",
    lm_config: Optional[LMConfig] = None
) -> Optional[LMEnhancedDecoder]:
    """
    Complete LM setup: train KenLM and build decoder.
    
    Args:
        processor: Wav2Vec2Processor for the model
        train_transcriptions: Training text to build LM from
        output_dir: Directory to save LM files
        lm_config: LM configuration
    
    Returns:
        LMEnhancedDecoder ready for CTC decoding, or None if setup fails
    """
    config = lm_config or LMConfig()
    
    if not config.train_kenlm:
        print("LM fusion disabled in config")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = LanguageModelTrainer(config)
    
    # Step 1: Prepare corpus
    corpus_path = f"{output_dir}/twi_corpus.txt"
    trainer.prepare_text_corpus(train_transcriptions, corpus_path)
    
    # Step 2: Train KenLM
    lm_binary_path = f"{output_dir}/twi_lm_{config.kenlm_order}gram.binary"
    
    if not Path(lm_binary_path).exists():
        lm_path = trainer.train_kenlm(corpus_path, lm_binary_path)
        if not lm_path:
            print("WARNING: LM training failed, continuing without LM fusion")
            return None
    else:
        print(f"Using existing LM: {lm_binary_path}")
        lm_path = lm_binary_path
    
    # Step 3: Build decoder
    vocab = trainer.build_vocabulary(train_transcriptions)
    decoder = LMEnhancedDecoder(
        processor=processor,
        kenlm_path=lm_path,
        vocab=vocab,
        alpha=config.lm_alpha,
        beta=config.lm_beta
    )
    
    return decoder


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    # Test with sample Twi text
    sample_texts = [
        "me pɛ sɛ me kɔ akwaaba",
        "ɔkɔɔ kurom no mu",
        "w'ani agye me",
        "mepɛ sɛ me hyɛ wo asɛm no nkyerɛ wo",
        "yɛbɛsane aba bio",
    ]
    
    print("Testing Language Model pipeline...")
    
    # Test corpus preparation
    trainer = LanguageModelTrainer()
    corpus_path = "/tmp/twi_test_corpus.txt"
    trainer.prepare_text_corpus(sample_texts, corpus_path)
    
    # Test vocabulary building
    vocab = trainer.build_vocabulary(sample_texts)
    print(f"\nVocabulary ({len(vocab)} words): {vocab[:10]}...")
    
    print("\nLM pipeline test complete!")
