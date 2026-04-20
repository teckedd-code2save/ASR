"""
ReachyAI ASR Data Pipeline
===========================
Preprocessing pipeline with disk caching for Akan/Twi ASR training.
Handles WaxalNLP, Farmerline, and Common Voice (MDC) datasets.
"""

import os
import re
import json
import shutil
import tarfile
import unicodedata
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from datasets import (
    Dataset, DatasetDict, Audio, load_dataset,
    load_from_disk, concatenate_datasets, IterableDataset
)

from config import DataConfig, setup_environment

# ====================================================================
# TEXT NORMALIZATION FOR AKAN/TWI
# ====================================================================

# Characters to remove (keep linguistically meaningful ones)
CHARS_TO_REMOVE = r'[\,\?\.\!\;\:\"\%\\\(\)\[\]\{\}]'

# Characters to KEEP in Twi: ɛ, ɔ, ŋ, apostrophe (elision), hyphen
# Do NOT remove: ', - (they carry linguistic meaning in Twi)

def normalize_text(text: str) -> str:
    """
    Normalize Twi/Akan text for ASR training.
    Preserves linguistically significant characters (ɛ, ɔ, ŋ, ', -).
    """
    if not text:
        return ""
    
    # Unicode NFC normalization (precomposed form)
    text = unicodedata.normalize('NFC', text)
    
    # Standardize quote variants to ASCII
    text = re.sub(r'[\u2018\u2019\u201a\u201b]', "'", text)
    text = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', text)
    
    # Remove only characters with no linguistic function
    text = re.sub(CHARS_TO_REMOVE, '', text)
    
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower().strip()


def compute_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    info = sf.info(audio_path)
    return info.duration


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sampling rate."""
    if orig_sr == target_sr:
        return audio_array
    return librosa.resample(audio_array.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr)


# ====================================================================
# DATASET LOADING
# ====================================================================

class DataPipeline:
    """End-to-end data loading and preprocessing pipeline with caching."""
    CACHE_SCHEMA_VERSION = "v2"
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.preprocessed_dir = Path(self.config.preprocessed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute cache key from config
        self.config_hash = self._compute_config_hash()
    
    def _compute_config_hash(self) -> str:
        """Compute a hash for cache invalidation."""
        config_dict = asdict(self.config)
        # Exclude paths that change between runs
        for key in ['cache_dir', 'preprocessed_dir']:
            config_dict.pop(key, None)
        config_dict["_cache_schema_version"] = self.CACHE_SCHEMA_VERSION
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        import hashlib
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_cache_path(self, name: str) -> Path:
        """Get cached dataset path."""
        return self.preprocessed_dir / f"{name}_{self.config_hash}"
    
    def _is_cached(self, name: str) -> bool:
        """Check if a preprocessed dataset exists in cache."""
        cache_path = self._get_cache_path(name)
        return cache_path.exists() and (cache_path / "dataset_info.json").exists()
    
    def _load_cached(self, name: str) -> Dataset:
        """Load a preprocessed dataset from cache."""
        cache_path = self._get_cache_path(name)
        print(f"  Loading cached: {cache_path}")
        return load_from_disk(str(cache_path))
    
    def _save_to_cache(self, dataset: Dataset, name: str) -> None:
        """Save a preprocessed dataset to cache."""
        cache_path = self._get_cache_path(name)
        print(f"  Caching to: {cache_path}")
        dataset.save_to_disk(str(cache_path))
    
    # ----------------------------------------------------------------
    # WaxalNLP Loading
    # ----------------------------------------------------------------
    
    def load_waxalnlp(self, split: str = "train") -> Dataset:
        """Load WaxalNLP Akan ASR dataset."""
        cache_key = f"waxal_{split}_raw"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            print(f"  Loading WaxalNLP {split} from local cache...")
            return load_from_disk(str(cache_path))
        
        print(f"  Downloading WaxalNLP {split} from HuggingFace...")
        ds = load_dataset(
            self.config.waxal_dataset,
            self.config.waxal_config,
            split=split,
            trust_remote_code=True
        )
        
        # Select only needed columns
        keep_cols = ["audio", "transcription"]
        available_cols = ds.column_names
        cols_to_remove = [c for c in available_cols if c not in keep_cols]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        
        # Cast audio to target sampling rate
        ds = ds.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        
        # Save raw cache
        ds.save_to_disk(str(cache_path))
        return ds
    
    # ----------------------------------------------------------------
    # Farmerline Loading
    # ----------------------------------------------------------------
    
    def load_farmerline(self, split: str = "train") -> Dataset:
        """Load Farmerline Twi dataset."""
        cache_key = f"farmerline_{split}_raw"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            print(f"  Loading Farmerline {split} from local cache...")
            return load_from_disk(str(cache_path))
        
        print(f"  Downloading Farmerline {split} from HuggingFace...")
        ds = load_dataset(
            self.config.farmerline_dataset,
            split=split,
            trust_remote_code=True
        )
        
        # Standardize column names
        if "sentence" in ds.column_names and "transcription" not in ds.column_names:
            ds = ds.rename_column("sentence", "transcription")
        
        keep_cols = ["audio", "transcription"]
        available_cols = ds.column_names
        cols_to_remove = [c for c in available_cols if c not in keep_cols]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        
        ds = ds.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        ds.save_to_disk(str(cache_path))
        return ds
    
    # ----------------------------------------------------------------
    # Common Voice Loading (from Mozilla Data Collective)
    # ----------------------------------------------------------------
    
    def load_common_voice(
        self,
        cv_base_dir: str,
        split: str = "train"
    ) -> Optional[Dataset]:
        """
        Load Common Voice Twi from local download.
        
        Args:
            cv_base_dir: Path to extracted Common Voice corpus
                        (e.g., "/path/to/cv-corpus-24.0-2025-12-05/tw")
            split: "train" or "test"
        """
        cache_key = f"cv_{split}_raw"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            print(f"  Loading Common Voice {split} from local cache...")
            return load_from_disk(str(cache_path))
        
        base_path = Path(cv_base_dir)
        if not base_path.exists():
            print(f"  WARNING: Common Voice directory not found: {base_path}")
            print(f"  Download from Mozilla Data Collective and extract to this path.")
            return None
        
        tsv_file = base_path / f"{split}.tsv"
        clips_dir = base_path / "clips"
        
        if not tsv_file.exists():
            print(f"  WARNING: TSV file not found: {tsv_file}")
            return None
        
        print(f"  Loading Common Voice {split} from {base_path}...")
        df = pd.read_csv(tsv_file, sep="\t")
        
        # Build audio paths and rename sentence -> transcription
        df = df[["path", "sentence"]].rename(columns={"sentence": "transcription"})
        df["audio"] = df["path"].apply(lambda p: str(clips_dir / p))
        df = df[["audio", "transcription"]]
        
        # Remove rows with missing audio files
        df = df[df["audio"].apply(lambda p: Path(p).exists())]
        
        if len(df) == 0:
            print(f"  WARNING: No valid audio files found for {split}")
            return None
        
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        ds = ds.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        ds.save_to_disk(str(cache_path))
        
        print(f"  Loaded {len(ds)} clips from Common Voice {split}")
        return ds
    
    # ----------------------------------------------------------------
    # Combine Datasets
    # ----------------------------------------------------------------
    
    def combine_datasets(
        self,
        datasets: Dict[str, Dataset],
        split: str = "train"
    ) -> Dataset:
        """Combine multiple datasets for a given split with source tracking."""
        valid_datasets = []
        for name, ds in datasets.items():
            if ds is not None and len(ds) > 0:
                # Add source column
                ds = ds.map(lambda x, n=name: {**x, "source": n}, batched=False)
                valid_datasets.append(ds)
                print(f"  {name}: {len(ds)} samples")
        
        if not valid_datasets:
            raise ValueError(f"No valid datasets for split '{split}'")
        
        combined = concatenate_datasets(valid_datasets)
        print(f"  Combined {split}: {len(combined)} total samples")
        return combined
    
    # ----------------------------------------------------------------
    # Preprocessing (Audio -> Features)
    # ----------------------------------------------------------------
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        processor: Any,  # Can be WhisperProcessor, Wav2Vec2Processor, etc.
        split_name: str,
        model_family: str = "omnilingual"
    ) -> Dataset:
        """
        Preprocess dataset: audio -> model inputs with disk caching.
        
        This is the expensive step that we cache aggressively.
        """
        cache_name = f"{split_name}_{model_family}_preprocessed"
        
        if self._is_cached(cache_name):
            print(f"  Using cached preprocessed: {cache_name}")
            return self._load_cached(cache_name)
        
        print(f"  Preprocessing {split_name} for {model_family}...")
        
        # Get feature extractor and tokenizer from processor
        if hasattr(processor, 'feature_extractor'):
            feature_extractor = processor.feature_extractor
        else:
            feature_extractor = processor
        
        has_tokenizer = hasattr(processor, 'tokenizer')
        tokenizer = processor.tokenizer if has_tokenizer else None
        
        def _preprocess_batch(batch):
            """Process a batch of audio samples."""
            # Process audio
            audio_arrays = []
            
            for audio in batch["audio"]:
                arr = audio["array"]
                sr = audio["sampling_rate"]
                
                # Resample if needed
                if sr != self.config.sampling_rate:
                    arr = librosa.resample(
                        arr.astype(np.float64),
                        orig_sr=sr,
                        target_sr=self.config.sampling_rate
                    )
                audio_arrays.append(arr)
            
            # Extract features based on model family
            if model_family in ["omnilingual", "mms", "xlsr", "w2vbert"]:
                # Wav2Vec2-style: raw audio -> CNN feature extractor
                inputs = feature_extractor(
                    audio_arrays,
                    sampling_rate=self.config.sampling_rate,
                    return_attention_mask=True,
                    padding=True,
                    return_tensors=None
                )
                batch["input_values"] = inputs["input_values"]
                if "attention_mask" in inputs:
                    batch["attention_mask"] = inputs["attention_mask"]
            else:
                # Whisper-style: log-mel spectrogram
                inputs = feature_extractor(
                    audio_arrays,
                    sampling_rate=self.config.sampling_rate,
                    return_tensors=None
                )
                batch["input_features"] = [f for f in inputs["input_features"]]
            
            # Tokenize transcriptions if tokenizer available
            if tokenizer is not None:
                texts = [normalize_text(t) for t in batch["transcription"]]
                labels = tokenizer(
                    texts,
                    return_attention_mask=False
                )
                batch["labels"] = labels["input_ids"]
                batch["transcription"] = texts

            if "source" in batch:
                batch["source"] = batch["source"]
            
            return batch
        
        # Process with batching for speed
        processed = dataset.map(
            _preprocess_batch,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc=f"Preprocessing {split_name}",
            num_proc=min(4, os.cpu_count() or 1)
        )
        
        # Cache the result
        self._save_to_cache(processed, cache_name)
        return processed
    
    # ----------------------------------------------------------------
    # Train/Validation Split
    # ----------------------------------------------------------------
    
    def create_train_val_split(
        self,
        train_dataset: Dataset,
        val_ratio: Optional[float] = None
    ) -> Tuple[Dataset, Dataset]:
        """Create train/validation split from training data."""
        val_ratio = val_ratio or self.config.val_ratio
        
        # Stratified split by source to maintain domain balance
        if "source" in train_dataset.column_names:
            sources = list(set(train_dataset["source"]))
            train_splits = []
            val_splits = []
            
            for source in sources:
                source_ds = train_dataset.filter(lambda x, s=source: x["source"] == s)
                split = source_ds.train_test_split(test_size=val_ratio, seed=42)
                train_splits.append(split["train"])
                val_splits.append(split["test"])
            
            train = concatenate_datasets(train_splits).shuffle(seed=42)
            val = concatenate_datasets(val_splits).shuffle(seed=42)
        else:
            split = train_dataset.train_test_split(test_size=val_ratio, seed=42)
            train = split["train"]
            val = split["test"]
        
        print(f"  Train: {len(train)}, Validation: {len(val)}")
        return train, val
    
    # ----------------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------------
    
    def build_dataset(
        self,
        processor: Any,
        model_family: str = "omnilingual",
        cv_base_dir: Optional[str] = None,
        skip_preprocessing: bool = False
    ) -> DatasetDict:
        """
        Build complete dataset with all splits.
        
        Returns:
            DatasetDict with "train", "validation", "test" splits
        """
        print("=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)
        
        # Check cache for full dataset
        full_cache = self._get_cache_path(f"fulldataset_{model_family}")
        if full_cache.exists() and not skip_preprocessing:
            print("Loading full preprocessed dataset from cache...")
            return load_from_disk(str(full_cache))
        
        # Load individual datasets
        print("\n1. Loading WaxalNLP...")
        waxal_train = self.load_waxalnlp("train")
        waxal_test = self.load_waxalnlp("test")
        
        print("\n2. Loading Farmerline...")
        farmerline_train = self.load_farmerline("train")
        farmerline_test = self.load_farmerline("test")
        
        print("\n3. Loading Common Voice...")
        cv_train = None
        cv_test = None
        if cv_base_dir:
            cv_train = self.load_common_voice(cv_base_dir, "train")
            cv_test = self.load_common_voice(cv_base_dir, "test")
        
        # Combine
        print("\n4. Combining datasets...")
        combined_train = self.combine_datasets({
            "waxal": waxal_train,
            "farmerline": farmerline_train,
            "common_voice": cv_train
        }, "train")
        
        combined_test = self.combine_datasets({
            "waxal": waxal_test,
            "farmerline": farmerline_test,
            "common_voice": cv_test
        }, "test")
        
        # Create validation split
        print("\n5. Creating train/validation split...")
        train, validation = self.create_train_val_split(combined_train)
        
        # Preprocess each split
        if not skip_preprocessing:
            print("\n6. Preprocessing train split...")
            train = self.preprocess_dataset(train, processor, "train", model_family)
            
            print("\n7. Preprocessing validation split...")
            validation = self.preprocess_dataset(validation, processor, "validation", model_family)
            
            print("\n8. Preprocessing test split...")
            test = self.preprocess_dataset(combined_test, processor, "test", model_family)
        else:
            test = combined_test
        
        dataset_dict = DatasetDict({
            "train": train,
            "validation": validation,
            "test": test
        })
        
        if not skip_preprocessing:
            print(f"\n9. Caching full dataset to {full_cache}...")
            dataset_dict.save_to_disk(str(full_cache))
        
        print("\n" + "=" * 60)
        print("DATASET BUILD COMPLETE")
        print(f"  Train:      {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        print(f"  Test:       {len(dataset_dict['test'])} samples")
        print("=" * 60)
        
        return dataset_dict


# ====================================================================
# PSEUDO-LABELING PIPELINE
# ====================================================================

def generate_pseudo_labels(
    model: Any,
    processor: Any,
    unlabeled_dataset: Dataset,
    config: DataConfig,
    device: str = "cuda",
    batch_size: int = 8
) -> Dataset:
    """
    Generate pseudo-labels for unlabeled audio using a trained model.
    Filters by confidence threshold.
    
    Returns dataset of (audio, pseudo_transcription) pairs.
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    model.to(device)
    
    max_clips = min(len(unlabeled_dataset), config.pseudo_label_max_clips)
    unlabeled_subset = unlabeled_dataset.select(range(max_clips))
    pseudo_labeled = []
    blank_token_id = getattr(model.config, "pad_token_id", None)
    if blank_token_id is None and hasattr(processor, "tokenizer"):
        blank_token_id = processor.tokenizer.pad_token_id
    
    # Prepare dataloader
    def _collate(batch):
        audio_arrays = [b["audio"]["array"] for b in batch]
        inputs = processor(
            audio_arrays,
            sampling_rate=config.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        return inputs
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        collate_fn=_collate
    )
    
    print(f"Generating pseudo-labels for {len(unlabeled_subset)} clips...")
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(loader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(model, "generate") and not hasattr(model, "lm_head"):
                outputs = model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=225
                )
                scores = torch.stack(outputs.scores, dim=1).softmax(-1)
                token_confidences = scores.max(-1).values
                transcriptions = processor.batch_decode(
                    outputs.sequences,
                    skip_special_tokens=True
                )
            else:
                logits = model(**inputs).logits
                probs = logits.softmax(dim=-1)
                pred_ids = torch.argmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values
                token_confidences = []

                for row_idx in range(pred_ids.shape[0]):
                    token_ids = pred_ids[row_idx]
                    conf_row = max_probs[row_idx]
                    valid_mask = torch.ones_like(token_ids, dtype=torch.bool)

                    if blank_token_id is not None:
                        valid_mask &= token_ids.ne(blank_token_id)

                    repeated = torch.zeros_like(valid_mask)
                    repeated[1:] = token_ids[1:] == token_ids[:-1]
                    valid_mask &= ~repeated

                    if "attention_mask" in inputs:
                        valid_mask &= inputs["attention_mask"][row_idx].bool()

                    if valid_mask.any():
                        token_confidences.append(conf_row[valid_mask])
                    else:
                        token_confidences.append(conf_row[:1])

                transcriptions = processor.batch_decode(pred_ids)
            
            # Filter by confidence
            for i, (text, conf) in enumerate(zip(transcriptions, token_confidences)):
                avg_conf = conf.mean().item()
                if avg_conf >= config.pseudo_label_confidence_threshold and text.strip():
                    idx = batch_idx * batch_size + i
                    pseudo_labeled.append({
                        "audio": unlabeled_subset[idx]["audio"],
                        "transcription": normalize_text(text),
                        "confidence": avg_conf,
                        "source": "pseudo_labeled"
                    })
    
    if not pseudo_labeled:
        print("WARNING: No pseudo-labels passed confidence threshold!")
        return Dataset.from_list([])
    
    pseudo_ds = Dataset.from_list(pseudo_labeled)
    print(f"Generated {len(pseudo_ds)} pseudo-labeled samples "
          f"(threshold: {config.pseudo_label_confidence_threshold})")
    
    return pseudo_ds


# ====================================================================
# MAIN ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    setup_environment()
    
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    print("Data pipeline initialized.")
    print(f"Cache dir: {config.cache_dir}")
    print(f"Config hash: {pipeline.config_hash}")
    
    # Quick test: load WaxalNLP
    print("\nTesting WaxalNLP loading...")
    waxal = pipeline.load_waxalnlp("train")
    print(f"WaxalNLP train: {len(waxal)} samples")
    print(f"Columns: {waxal.column_names}")
    print(f"Sample: {waxal[0]}")
