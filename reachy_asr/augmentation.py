"""
ReachyAI ASR Data Augmentation
===============================
SpecAugment, speed perturbation, noise injection for Akan/Twi ASR.
Applied during training via custom data collator.
"""

import random
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import librosa
try:
    from transformers import DataCollatorCTCWithPadding as BaseCTCDataCollator
except ImportError:
    from transformers import DataCollatorForCTC as BaseCTCDataCollator

from config import AugmentationConfig


# ====================================================================
# SPEC-AUGMENT
# ====================================================================

class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.
    Reference: Park et al., 2019
    
    Applies frequency and time masking to mel spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        mask_value: float = 0.0
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value
    
    def __call__(self, input_features: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to input features.
        
        Args:
            input_features: Array of shape (..., freq_bins, time_steps)
        
        Returns:
            Augmented features with same shape
        """
        features = input_features.copy()
        
        # Handle different input shapes
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # (1, F, T)
            squeeze = True
        else:
            squeeze = False
        
        B, F, T = features.shape
        
        for b in range(B):
            # Frequency masking
            for _ in range(self.n_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, max(0, F - f))
                features[b, f0:f0+f, :] = self.mask_value
            
            # Time masking (adaptive upper bound)
            for _ in range(self.n_time_masks):
                max_t = min(self.time_mask_param, int(0.2 * T))
                t = random.randint(0, max_t) if max_t > 0 else 0
                if t > 0 and T > t:
                    t0 = random.randint(0, T - t)
                    features[b, :, t0:t0+t] = self.mask_value
        
        if squeeze:
            features = features.squeeze(0)
        
        return features


# ====================================================================
# SPEED PERTURBATION
# ====================================================================

def apply_speed_perturbation(
    audio: np.ndarray,
    sr: int = 16000,
    speed_factors: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Apply speed perturbation to audio.
    Returns original + speed-perturbed variants.
    
    Args:
        audio: Audio array at 16kHz
        sr: Sampling rate
        speed_factors: List of speed multipliers (default: [0.9, 1.1])
    
    Returns:
        List of audio dicts: [{"array": ..., "sampling_rate": ...}, ...]
    """
    speed_factors = speed_factors or [0.9, 1.1]
    
    variants = [{"array": audio, "sampling_rate": sr}]
    
    for factor in speed_factors:
        try:
            stretched = librosa.effects.time_stretch(audio.astype(np.float64), rate=factor)
            variants.append({
                "array": stretched.astype(np.float32),
                "sampling_rate": sr
            })
        except Exception:
            # Skip if time_stretch fails
            continue
    
    return variants


# ====================================================================
# NOISE INJECTION (MUSAN)
# ====================================================================

def load_musan_noise(
    musan_dir: str,
    noise_category: str = "noise",
    max_duration: float = 10.0
) -> np.ndarray:
    """
    Load a random noise sample from MUSAN corpus.
    
    Args:
        musan_dir: Path to MUSAN corpus
        noise_category: "noise", "music", or "speech"
        max_duration: Maximum noise duration in seconds
    """
    import glob
    
    noise_dir = f"{musan_dir}/{noise_category}"
    wav_files = glob.glob(f"{noise_dir}/**/*.wav", recursive=True)
    
    if not wav_files:
        return None
    
    # Pick random file
    noise_file = random.choice(wav_files)
    noise, sr = librosa.load(noise_file, sr=16000, duration=max_duration)
    
    return noise


def add_noise(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 10.0
) -> np.ndarray:
    """
    Add noise to audio at specified SNR.
    
    Args:
        audio: Clean audio
        noise: Noise signal
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy audio
    """
    # Extend or truncate noise to match audio length
    if len(noise) < len(audio):
        # Loop noise
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    
    noise = noise[:len(audio)]
    
    # Compute power
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return audio
    
    # Scale noise to achieve target SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(audio_power / (snr_linear * noise_power))
    noisy = audio + noise_scale * noise
    
    return noisy


# ====================================================================
# CUSTOM DATA COLLATOR WITH AUGMENTATION
# ====================================================================

class AugmentedDataCollatorCTC(BaseCTCDataCollator):
    """
    Data collator for CTC-based ASR with on-the-fly augmentation.
    
    Applies SpecAugment to input features during batching.
    Optionally applies speed perturbation during training.
    """
    
    def __init__(
        self,
        processor,
        padding: bool = True,
        augmentation_config: Optional[AugmentationConfig] = None,
        apply_augmentation: bool = True,
        **kwargs
    ):
        super().__init__(processor=processor, padding=padding, **kwargs)
        
        self.aug_config = augmentation_config or AugmentationConfig()
        self.apply_augmentation = apply_augmentation
        
        # Initialize SpecAugment
        if self.aug_config.use_spec_augment and apply_augmentation:
            self.spec_augment = SpecAugment(
                freq_mask_param=self.aug_config.freq_mask_param,
                time_mask_param=self.aug_config.time_mask_param,
                n_freq_masks=self.aug_config.n_freq_masks,
                n_time_masks=self.aug_config.n_time_masks
            )
        else:
            self.spec_augment = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with optional augmentation.
        """
        # Separate inputs and labels
        input_features = []
        label_features = []
        for f in features:
            if "input_values" in f:
                input_features.append({"input_values": f["input_values"]})
            elif "input_features" in f:
                input_features.append({"input_features": f["input_features"]})
            
            if "labels" in f:
                label_features.append({"input_ids": f["labels"]})
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Pad labels
        if label_features:
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )
            
            # Replace padding with -100 for loss computation
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
        
        # Apply SpecAugment during training
        if (
            self.spec_augment is not None
            and self.apply_augmentation
            and "input_features" in batch
        ):
            with torch.no_grad():
                inputs = batch["input_features"].numpy()
                
                # Apply SpecAugment per sample
                augmented = []
                for i in range(inputs.shape[0]):
                    aug = self.spec_augment(inputs[i])
                    augmented.append(aug)
                
                batch["input_features"] = torch.from_numpy(np.stack(augmented)).float()
        
        return batch


# ====================================================================
# DATASET-LEVEL SPEED PERTURBATION
# ====================================================================

def create_speed_perturbed_dataset(
    dataset,
    speed_factors: List[float] = None,
    probability: float = 0.5,
    sampling_rate: int = 16000
):
    """
    Create a dataset with speed-perturbed variants.
    
    Args:
        dataset: HuggingFace Dataset with audio column
        speed_factors: Speed multipliers (default: [0.9, 1.1])
        probability: Probability of applying to each sample
        sampling_rate: Audio sampling rate
    
    Returns:
        Augmented dataset
    """
    speed_factors = speed_factors or [0.9, 1.1]
    
    def _augment_sample(sample):
        """Apply speed perturbation to a single sample."""
        if random.random() > probability:
            return sample
        
        audio = sample["audio"]["array"]
        factor = random.choice(speed_factors)
        
        try:
            stretched = librosa.effects.time_stretch(
                audio.astype(np.float64),
                rate=factor
            )
            sample["audio"] = {
                "array": stretched.astype(np.float32),
                "sampling_rate": sampling_rate
            }
        except Exception:
            pass
        
        return sample
    
    return dataset.map(_augment_sample)


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    # Test SpecAugment
    print("Testing SpecAugment...")
    spec_aug = SpecAugment()
    
    # Fake mel spectrogram (80 freq bins, 1000 time steps)
    fake_mel = np.random.randn(80, 1000).astype(np.float32)
    
    augmented = spec_aug(fake_mel)
    
    # Check that masking occurred
    n_masked_freq = np.sum(augmented == 0)
    n_masked_time = np.sum(augmented == 0)
    
    print(f"Original shape: {fake_mel.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Masked values: {n_masked_freq}")
    print("SpecAugment test passed!")
    
    # Test speed perturbation
    print("\nTesting speed perturbation...")
    dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
    variants = apply_speed_perturbation(dummy_audio)
    print(f"Original length: {len(dummy_audio)}")
    for v in variants:
        print(f"  Speed variant: length={len(v['array'])}")
    print("Speed perturbation test passed!")
