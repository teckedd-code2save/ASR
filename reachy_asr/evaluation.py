"""
ReachyAI ASR Evaluation
=======================
Comprehensive evaluation with proper normalization for Akan/Twi:
- WER with text normalization (comparable to benchmarks)
- CER (Character Error Rate) for diacritic analysis
- Per-domain evaluation (Waxal vs Farmerline vs Common Voice)
- Error analysis for Twi-specific characters (ɛ, ɔ, ŋ, tone markers)
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
from jiwer import wer as compute_wer, cer as compute_cer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from config import EvalConfig


# ====================================================================
# TEXT NORMALIZATION
# ====================================================================

class TwiTextNormalizer:
    """
    Text normalizer for Twi/Akan ASR evaluation.
    
    Uses Whisper's BasicTextNormalizer as base, with Twi-specific
    adjustments to preserve linguistically meaningful characters.
    """
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.whisper_normalizer = BasicTextNormalizer() if self.config.use_whisper_normalizer else None
    
    def normalize(self, text: str) -> str:
        """Apply normalization pipeline."""
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Step 2: Whisper's built-in normalization (if enabled)
        if self.whisper_normalizer:
            text = self.whisper_normalizer(text)
        
        # Step 3: Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Step 4: Custom Twi normalization
        # Standardize apostrophes (elision marker in Twi)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        
        # Step 5: Remove punctuation (keep linguistically significant ones)
        if self.config.remove_punctuation:
            # Keep: ɛ ɔ ŋ ' - (linguistically significant)
            # Remove: . , ? ! ; : " % ( ) [ ] { } etc.
            text = re.sub(r'[\.\,\?\!\;\:\"\%\(\)\[\]\{\}]', '', text)
        
        # Step 6: Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]


# ====================================================================
# METRICS COMPUTATION
# ====================================================================

class ASREvaluator:
    """
    Comprehensive ASR evaluator with Twi-specific analysis.
    """
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.normalizer = TwiTextNormalizer(self.config)
    
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute standard ASR metrics with proper normalization.
        
        Args:
            predictions: Raw model outputs
            references: Ground truth transcriptions
        
        Returns:
            Dictionary of metrics
        """
        assert len(predictions) == len(references), "Length mismatch"
        
        # Normalize both predictions and references
        norm_preds = self.normalizer.normalize_batch(predictions)
        norm_refs = self.normalizer.normalize_batch(references)
        
        # Filter empty pairs
        valid_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if r.strip()]
        if not valid_pairs:
            return {"wer": 100.0, "cer": 100.0}
        
        norm_preds, norm_refs = zip(*valid_pairs)
        
        results = {}
        
        # WER
        if self.config.compute_wer:
            results["wer"] = compute_wer(norm_refs, norm_preds) * 100
        
        # CER
        if self.config.compute_cer:
            results["cer"] = compute_cer(norm_refs, norm_preds) * 100
        
        return results
    
    def compute_per_domain_metrics(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per data source (Waxal, Farmerline, Common Voice).
        
        Returns:
            {source_name: {metric: value}}
        """
        results = {}
        
        # Group by source
        source_groups = defaultdict(lambda: {"preds": [], "refs": []})
        for pred, ref, src in zip(predictions, references, sources):
            source_groups[src]["preds"].append(pred)
            source_groups[src]["refs"].append(ref)
        
        # Compute per-source metrics
        for src, group in source_groups.items():
            src_metrics = self.compute_metrics(group["preds"], group["refs"])
            results[src] = src_metrics
        
        # Also compute overall
        results["overall"] = self.compute_metrics(predictions, references)
        
        return results
    
    def compute_diacritic_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Analyze errors on Twi-specific diacritics and characters.
        
        Tracks:
        - ɛ (open e) errors
        - ɔ (open o) errors  
        - ŋ (eng) errors
        - Tone marker preservation
        """
        twi_chars = ['ɛ', 'ɔ', 'ŋ', 'Ɛ', 'Ɔ', 'Ŋ']
        
        char_total = {c: 0 for c in twi_chars}
        char_correct = {c: 0 for c in twi_chars}
        
        for pred, ref in zip(predictions, references):
            for c in twi_chars:
                count_ref = ref.count(c)
                count_pred = pred.count(c)
                char_total[c] += count_ref
                # Approximate: correct if count matches
                char_correct[c] += min(count_ref, count_pred)
        
        results = {}
        for c in twi_chars:
            if char_total[c] > 0:
                accuracy = 100 * char_correct[c] / char_total[c]
                results[f"{c}_accuracy"] = accuracy
                results[f"{c}_total"] = char_total[c]
        
        return results
    
    def error_analysis(
        self,
        predictions: List[str],
        references: List[str],
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Detailed error analysis.
        
        Returns:
            Dictionary with error statistics and examples.
        """
        norm_preds = self.normalizer.normalize_batch(predictions)
        norm_refs = self.normalizer.normalize_batch(references)
        
        errors = []
        for i, (pred, ref) in enumerate(zip(norm_preds, norm_refs)):
            sample_wer = compute_wer([ref], [pred]) * 100 if ref.strip() else 100
            if sample_wer > 0:
                errors.append({
                    "index": i,
                    "reference": ref,
                    "prediction": pred,
                    "wer": sample_wer
                })
        
        # Sort by WER descending
        errors.sort(key=lambda x: x["wer"], reverse=True)
        
        # Common error patterns
        all_ref_words = " ".join(norm_refs).split()
        all_pred_words = " ".join(norm_preds).split()
        
        ref_counter = Counter(all_ref_words)
        pred_counter = Counter(all_pred_words)
        
        # Words that appear in references but not in predictions (deletions)
        missing_words = []
        for word, count in ref_counter.most_common(100):
            if word not in pred_counter or pred_counter[word] < count:
                missing_words.append({
                    "word": word,
                    "ref_count": count,
                    "pred_count": pred_counter.get(word, 0)
                })
        
        return {
            "total_samples": len(predictions),
            "error_samples": len(errors),
            "worst_errors": errors[:top_n],
            "most_missing_words": missing_words[:top_n],
            "diacritic_metrics": self.compute_diacritic_metrics(norm_preds, norm_refs)
        }


# ====================================================================
# TRAINER INTEGRATION
# ====================================================================

def create_compute_metrics_fn(processor, config: Optional[EvalConfig] = None):
    """
    Create a compute_metrics function compatible with HuggingFace Trainer.
    
    Usage:
        trainer = Trainer(
            ...,
            compute_metrics=create_compute_metrics_fn(processor)
        )
    """
    evaluator = ASREvaluator(config)
    
    def compute_metrics(pred) -> Dict[str, float]:
        """
        Compute metrics from Trainer prediction output.
        
        pred.predictions: Model logits or generated token IDs
        pred.label_ids: Ground truth labels (with -100 padding)
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Handle different prediction formats
        if hasattr(pred_ids, 'shape') and len(pred_ids.shape) == 3:
            # Logits (seq2seq) - take argmax
            pred_ids = np.argmax(pred_ids, axis=-1)
        
        # Replace -100 with pad_token_id for decoding
        if hasattr(label_ids, 'shape'):
            label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        
        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute metrics
        metrics = evaluator.compute_metrics(pred_str, label_str)
        
        return metrics
    
    return compute_metrics


# ====================================================================
# TEST
# ====================================================================

if __name__ == "__main__":
    evaluator = ASREvaluator()
    
    # Test with Twi text
    predictions = [
        "me pɛ sɛ me kɔ akwaaba",
        "ɔkɔɔ kurom no mu",
        "w'ani agye me",
    ]
    references = [
        "mepɛ sɛ me kɔ akwaaba",
        "ɔkɔɔ kurom no mu",
        "wo nani agye me",
    ]
    
    metrics = evaluator.compute_metrics(predictions, references)
    print("Metrics:", metrics)
    
    # Test diacritic analysis
    diacritics = evaluator.compute_diacritic_metrics(predictions, references)
    print("Diacritics:", diacritics)
    
    # Test normalizer
    normalizer = TwiTextNormalizer()
    test_text = "Gɛls fɔr egyina ha obiaa kita buk ɔmo gyina pila kɛseɛ be ho baako abɔ ne tii tententen ."
    normalized = normalizer.normalize(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Normalized: {normalized}")
