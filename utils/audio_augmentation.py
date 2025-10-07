import os
import random
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import soundfile


class AudioAugmenter:
    """
    Audio augmentation class for applying various transformations to audio samples.
    """

    def __init__(self, augment_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the AudioAugmenter.

        Args:
            augment_configs: List of augmentation configurations, each containing:
                - type: Type of augmentation ('speed', 'shift', 'volume', 'resample', 'noise', 'gaussian')
                - prob: Probability of applying this augmentation
                - params: Parameters specific to each augmentation type
        """
        self.augment_configs = augment_configs or []
        self.speed_rates = None
        self.noises_path = None

    def augment(self, sample: np.ndarray, sample_rate: int) -> tuple:
        """
        Apply configured augmentations to an audio sample.

        Args:
            sample: Audio sample as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (augmented_sample, sample_rate)
        """
        for config in self.augment_configs:
            if config["type"] == "speed" and random.random() < config["prob"]:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = (
                        config["params"]["min_speed_rate"],
                        config["params"]["max_speed_rate"],
                        config["params"]["num_rates"],
                    )
                    self.speed_rates = np.linspace(
                        min_speed_rate, max_speed_rate, num_rates, endpoint=True
                    )
                rate = random.choice(self.speed_rates)
                sample = change_speed(sample, speed_rate=rate)

            if config["type"] == "shift" and random.random() < config["prob"]:
                min_shift_ms, max_shift_ms = (
                    config["params"]["min_shift_ms"],
                    config["params"]["max_shift_ms"],
                )
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = shift(sample, sample_rate, shift_ms=shift_ms)

            if config["type"] == "volume" and random.random() < config["prob"]:
                min_gain_dBFS, max_gain_dBFS = (
                    config["params"]["min_gain_dBFS"],
                    config["params"]["max_gain_dBFS"],
                )
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = volume(sample, gain=gain)

            if config["type"] == "resample" and random.random() < config["prob"]:
                new_sample_rates = config["params"]["new_sample_rates"]
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = resample(
                    sample, orig_sr=sample_rate, target_sr=new_sample_rate
                )
                sample_rate = new_sample_rate

            if config["type"] == "noise" and random.random() < config["prob"]:
                min_snr_dB, max_snr_dB = (
                    config["params"]["min_snr_dB"],
                    config["params"]["max_snr_dB"],
                )
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config["params"]["noise_dir"]
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                if self.noises_path:
                    noise_path = random.choice(self.noises_path)
                    snr_dB = random.randint(min_snr_dB, max_snr_dB)
                    sample = add_noise(
                        sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB
                    )

            if config["type"] == "gaussian" and random.random() < config["prob"]:
                min_snr_dB, max_snr_dB = (
                    config["params"]["min_snr_dB"],
                    config["params"]["max_snr_dB"],
                )
                snr_dB = random.uniform(min_snr_dB, max_snr_dB)
                sample = add_gaussian_noise(sample, snr_dB=snr_dB)

        return sample, sample_rate


def change_speed(sample: np.ndarray, speed_rate: float) -> np.ndarray:
    """
    Change speech speed.

    Args:
        sample: Audio sample as numpy array
        speed_rate: Speed change factor (1.0 = no change, <1.0 = slower, >1.0 = faster)

    Returns:
        Speed-adjusted audio sample
    """
    if speed_rate == 1.0:
        return sample
    if speed_rate <= 0:
        raise ValueError("Speed rate should be greater than zero")
    old_length = sample.shape[0]
    new_length = int(old_length / speed_rate)
    old_indices = np.arange(old_length)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)
    sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
    return sample


def shift(sample: np.ndarray, sample_rate: int, shift_ms: int) -> np.ndarray:
    """
    Shift audio in time.

    Args:
        sample: Audio sample as numpy array
        sample_rate: Sample rate of the audio
        shift_ms: Shift amount in milliseconds (positive = right shift, negative = left shift)

    Returns:
        Time-shifted audio sample
    """
    duration = sample.shape[0] / sample_rate
    if abs(shift_ms) / 1000.0 > duration:
        raise ValueError(
            "Absolute value of shift_ms should be less than audio duration"
        )
    shift_samples = int(shift_ms * sample_rate / 1000)
    if shift_samples > 0:
        sample[:-shift_samples] = sample[shift_samples:]
        sample[-shift_samples:] = 0
    elif shift_samples < 0:
        sample[-shift_samples:] = sample[:shift_samples]
        sample[:-shift_samples] = 0
    return sample


def volume(sample: np.ndarray, gain: float) -> np.ndarray:
    """
    Change volume by applying gain.

    Args:
        sample: Audio sample as numpy array
        gain: Gain in decibels

    Returns:
        Volume-adjusted audio sample
    """
    sample *= 10.0 ** (gain / 20.0)
    return sample


def resample(sample: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate (fast polyphase with safe fallback).

    Args:
        sample: Audio sample as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    sample = np.asarray(sample, dtype=np.float32)
    if orig_sr == target_sr:
        return sample
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(orig_sr), int(target_sr))
        up = target_sr // g
        down = orig_sr // g
        # sample expected to be 1D; ensure it
        if sample.ndim > 1:
            sample = sample.reshape(-1)
        out = resample_poly(sample, up, down)
        return out.astype(np.float32, copy=False)
    except Exception:
        # Fallback to librosa if SciPy isn't available
        return librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr).astype(
            np.float32, copy=False
        )


def add_noise(
    sample: np.ndarray,
    sample_rate: int,
    noise_path: str,
    snr_dB: float,
    max_gain_db: float = 300.0,
) -> np.ndarray:
    """
    Add noise from a file to the audio sample (fast: soundfile + cheap mono + our resampler).

    Args:
        sample: Audio sample as numpy array (1D float32 mono)
        sample_rate: Sample rate of the audio
        noise_path: Path to noise audio file
        snr_dB: Signal-to-noise ratio in decibels
        max_gain_db: Maximum gain in decibels

    Returns:
        Audio sample with added noise
    """
    # sample is 1D float32 mono
    noise, sr = soundfile.read(noise_path, dtype="float32", always_2d=True)
    # Cheap mono
    if noise.ndim == 2:
        if noise.shape[1] > 1:
            noise = noise.mean(axis=1).astype(np.float32)
        else:
            noise = noise[:, 0].astype(np.float32)
    else:
        noise = noise.astype(np.float32)

    # Resample noise if needed (use fast polyphase)
    if sr != sample_rate:
        noise = resample(noise, orig_sr=sr, target_sr=sample_rate)

    # Normalize audio volume to ensure noise is not too loud
    target_db = -20
    gain = min(max_gain_db, target_db - rms_db(sample))
    sample = (sample * (10.0 ** (gain / 20.0))).astype(np.float32)

    # Specify noise volume
    sample_rms_db, noise_rms_db = rms_db(sample), rms_db(noise)
    noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
    noise = (noise * (10.0 ** (noise_gain_db / 20.0))).astype(np.float32)

    # Fix noise length
    if noise.shape[0] < sample.shape[0]:
        rep = int(np.ceil(sample.shape[0] / noise.shape[0]))
        noise = np.tile(noise, rep)[: sample.shape[0]]
    elif noise.shape[0] > sample.shape[0]:
        start_frame = random.randint(0, noise.shape[0] - sample.shape[0])
        noise = noise[start_frame : start_frame + sample.shape[0]]

    sample = (sample + noise).astype(np.float32)
    return sample


def add_gaussian_noise(sample: np.ndarray, snr_dB: float) -> np.ndarray:
    """
    Add Gaussian white noise to audio sample.

    Args:
        sample: Audio signal as 1D numpy array
        snr_dB: Signal-to-noise ratio in decibels

    Returns:
        Noisy audio sample
    """
    # Calculate signal power and convert SNR from dB
    signal_power = np.mean(sample**2)
    snr_linear = 10 ** (snr_dB / 10)

    # Calculate noise power based on desired SNR
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), sample.shape)

    # Add noise to signal
    noisy_sample = (sample + noise).astype(np.float32)

    # Clip to prevent overflow (optional, but recommended for audio)
    noisy_sample = np.clip(noisy_sample, -1.0, 1.0)

    return noisy_sample


def rms_db(sample: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) in decibels.

    Args:
        sample: Audio sample as numpy array

    Returns:
        RMS value in decibels
    """
    mean_square = np.mean(sample**2) + 1e-12  # numerical safety
    return 10 * np.log10(mean_square)
