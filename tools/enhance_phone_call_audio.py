#!/usr/bin/env python3
"""Enhance phone-call audio before ASR inference.

This script is intentionally simple and dependency-light. It uses only numpy,
librosa, and soundfile, which are already in this repository requirements.
"""

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}


def import_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "This script requires librosa. Install it with: pip install librosa"
        ) from exc
    return librosa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input audio file or directory")
    parser.add_argument("output", type=str, help="Output file or directory")
    parser.add_argument(
        "--target-sr", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--band-low-hz",
        type=float,
        default=120.0,
        help="Lower bound of speech-focused band-pass",
    )
    parser.add_argument(
        "--band-high-hz",
        type=float,
        default=3800.0,
        help="Upper bound of speech-focused band-pass",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=1.4,
        help="Noise reduction strength (larger = more aggressive)",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=20.0,
        help="Percentile used to estimate stationary noise profile",
    )
    parser.add_argument(
        "--min-mask",
        type=float,
        default=0.08,
        help="Minimum retained spectral gain during denoise",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=-3.0,
        help="Target peak level in dBFS after processing",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading/trailing silence before enhancement",
    )
    parser.add_argument(
        "--trim-top-db",
        type=float,
        default=35.0,
        help="dB threshold for silence trimming when --trim-silence is enabled",
    )
    return parser.parse_args()


def discover_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    files = [
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def ensure_output_paths(
    inputs: Sequence[Path], input_root: Path, output_root: Path
) -> List[Path]:
    if input_root.is_file():
        return [output_root]

    output_paths: List[Path] = []
    for in_path in inputs:
        rel_path = in_path.relative_to(input_root)
        out_path = output_root / rel_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.append(out_path)
    return output_paths


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1).astype(np.float32, copy=False)


def fft_bandpass(
    samples: np.ndarray, sr: int, low_hz: float, high_hz: float
) -> np.ndarray:
    if low_hz <= 0 and high_hz >= sr / 2:
        return samples

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.shape[0], d=1.0 / sr)
    keep = (freqs >= max(0.0, low_hz)) & (freqs <= min(high_hz, sr / 2.0))
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=samples.shape[0]).astype(np.float32, copy=False)


def spectral_denoise(
    samples: np.ndarray,
    strength: float,
    noise_percentile: float,
    min_mask: float,
) -> np.ndarray:
    librosa = import_librosa()
    n_fft = 512
    hop = 128

    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_mag = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)
    noise_power = np.square(noise_mag)
    signal_power = np.square(magnitude)

    residual_power = np.maximum(signal_power - strength * noise_power, 0.0)
    mask = residual_power / (residual_power + strength * noise_power + 1e-8)
    mask = np.clip(mask, min_mask, 1.0)

    cleaned_stft = magnitude * mask * np.exp(1j * phase)
    cleaned = librosa.istft(
        cleaned_stft, hop_length=hop, win_length=n_fft, length=len(samples)
    )
    return cleaned.astype(np.float32, copy=False)


def peak_normalize(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak < 1e-8:
        return samples

    target_peak = 10 ** (target_dbfs / 20.0)
    gain = target_peak / peak
    normalized = samples * gain
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def enhance_audio(
    samples: np.ndarray,
    sr: int,
    target_sr: int,
    band_low_hz: float,
    band_high_hz: float,
    denoise_strength: float,
    noise_percentile: float,
    min_mask: float,
    target_dbfs: float,
    trim_silence: bool,
    trim_top_db: float,
) -> Tuple[np.ndarray, int]:
    librosa = import_librosa()
    mono = to_mono(samples)

    if trim_silence:
        mono, _ = librosa.effects.trim(mono, top_db=trim_top_db)

    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    filtered = fft_bandpass(mono, sr=sr, low_hz=band_low_hz, high_hz=band_high_hz)
    cleaned = spectral_denoise(
        filtered,
        strength=denoise_strength,
        noise_percentile=noise_percentile,
        min_mask=min_mask,
    )
    normalized = peak_normalize(cleaned, target_dbfs=target_dbfs)
    return normalized, sr


def process_file(in_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    samples, sr = sf.read(in_path, dtype="float32", always_2d=True)
    samples = samples.squeeze()

    enhanced, out_sr = enhance_audio(
        samples=samples,
        sr=sr,
        target_sr=args.target_sr,
        band_low_hz=args.band_low_hz,
        band_high_hz=args.band_high_hz,
        denoise_strength=args.denoise_strength,
        noise_percentile=args.noise_percentile,
        min_mask=args.min_mask,
        target_dbfs=args.target_dbfs,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, enhanced, out_sr, subtype="PCM_16")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_sr <= 0:
        raise ValueError("--target-sr must be > 0")
    if not (0.0 < args.noise_percentile < 100.0):
        raise ValueError("--noise-percentile must be between 0 and 100")
    if args.denoise_strength <= 0:
        raise ValueError("--denoise-strength must be > 0")
    if not (0.0 <= args.min_mask <= 1.0):
        raise ValueError("--min-mask must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_as_dir = args.output.endswith("/") or args.output.endswith("\\")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = discover_audio_files(input_path)
    if not files:
        raise FileNotFoundError(f"No audio files found at: {input_path}")

    if input_path.is_file():
        if output_path.is_dir() or output_as_dir:
            output_file = output_path / input_path.with_suffix(".wav").name
        else:
            output_file = output_path.with_suffix(".wav")
        targets = [output_file]
    else:
        if output_path.suffix:
            raise ValueError(
                "When input is a directory, output must be a directory path."
            )
        targets = ensure_output_paths(
            files, input_root=input_path, output_root=output_path
        )

    for in_path, out_path in zip(files, targets):
        process_file(in_path=in_path, out_path=out_path, args=args)
        print(f"Processed: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
