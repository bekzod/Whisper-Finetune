import csv
import json
import os
import random
import sys
from typing import List
import glob

import librosa
import numpy as np
import pandas as pd
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader


def remove_silence_librosa(
    y: np.ndarray,
    sr: int,
    top_db: int = 40,
    min_silence_ms: float = 200,
    pad_ms: float = 50,
) -> np.ndarray:
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        return np.array([], dtype=y.dtype)

    def merge_intervals(iv, min_gap):
        merged = []
        for start, end in iv:
            if not merged:
                merged.append([start, end])
                continue
            if start - merged[-1][1] <= min_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return np.array(merged, dtype=int)

    min_gap = int((min_silence_ms / 1000.0) * sr)
    intervals = merge_intervals(intervals, min_gap=min_gap)

    pad = int((pad_ms / 1000.0) * sr)
    chunks = []
    for s, e in intervals:
        s = max(0, s - pad)
        e = min(len(y), e + pad)
        chunks.append(y[s:e])
    return np.concatenate(chunks) if chunks else np.array([], dtype=y.dtype)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        processor,
        mono=True,
        language=None,
        timestamps=False,
        sample_rate=16000,
        min_duration=0.5,
        max_duration=30,
        min_sentence=1,
        max_sentence=200,
        augment_config_path=None,
        # --- optional knobs for Example A ---
        silence_top_db: int = 40,
        silence_min_gap_ms: int = 200,
        silence_pad_ms: int = 50,
    ):
        """
        Args:
            data_list_path: Path to the data list file, or path to the binary list header file.
                           Supports JSON, JSONL, CSV, and Parquet formats.
                           Multiple paths can be specified separated by '+' to combine datasets,
                           e.g., '../datasets/train.json+../datasets/cleaned.json'
                           CSV format supports:
                           - Standard format: filename,text
                           - LJSpeech format: filename|text
                           - Header detection for column names like 'filename', 'text', 'audio_path', etc.
            processor: Whisper preprocessing tool, obtained from WhisperProcessor.from_pretrained
            mono: Whether to convert audio to mono channel, this must be True
            language: Language of the fine-tuning data
            timestamps: Whether to use timestamps during fine-tuning
            sample_rate: Audio sample rate, default is 16000
            min_duration: Audio shorter than this duration will be truncated, in seconds, cannot be less than 0.5, default 0.5s
            max_duration: Audio longer than this duration will be truncated, in seconds, cannot be greater than 30, default 30s
            min_sentence: Minimum sentence character count for fine-tuning, default 1
            max_sentence: Maximum sentence character count for fine-tuning, default 200
            augment_config_path: Path to data augmentation configuration parameter file

            Example A params:
            silence_top_db: energy threshold for librosa.effects.split (smaller = more aggressive)
            silence_min_gap_ms: merge gaps shorter than this between voiced islands
            silence_pad_ms: pad each kept island to avoid cutting phones
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, (
            f"min_duration cannot be less than 0.5, current value: {min_duration}"
        )
        assert max_duration <= 30, (
            f"max_duration cannot be greater than 30, current value: {max_duration}"
        )
        assert min_sentence >= 1, (
            f"min_sentence cannot be less than 1, current value: {min_sentence}"
        )
        assert max_sentence <= 200, (
            f"max_sentence cannot be greater than 200, current value: {max_sentence}"
        )
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sentence = min_sentence
        self.max_sentence = max_sentence

        # Example A config
        self.silence_top_db = silence_top_db
        self.silence_min_gap_ms = silence_min_gap_ms
        self.silence_pad_ms = silence_pad_ms

        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab["<|startoftranscript|>"]
        self.endoftext = self.vocab["<|endoftext|>"]
        if "<|nospeech|>" in self.vocab.keys():
            self.nospeech = self.vocab["<|nospeech|>"]
            self.timestamp_begin = None
        else:
            # Compatible with old models
            self.nospeech = self.vocab["<|nocaptions|>"]
            self.timestamp_begin = self.vocab["<|notimestamps|>"] + 1

        self.data_list: List[dict] = []
        self._load_data_list()

        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, "r", encoding="utf-8") as f:
                self.augment_configs = json.load(f)

    # Load data list
    def _load_data_list(self):
        # Support multiple dataset paths separated by '+'
        data_paths = self.data_list_path.split("+")
        self.data_list = []

        for data_path in data_paths:
            data_path = data_path.strip()  # Remove any whitespace
            if data_path.endswith(".header"):
                # Get binary data list
                dataset_reader = DatasetReader(
                    data_header_path=data_path,
                    min_duration=self.min_duration,
                    max_duration=self.max_duration,
                )
                current_data_list = dataset_reader.get_keys()
                self.data_list.extend(current_data_list)
            elif data_path.endswith(".parquet"):
                # Load single parquet file
                df = pd.read_parquet(data_path)
                self._process_dataframe(df, data_path)
            elif "*" in data_path and "parquet" in data_path:
                # Load multiple parquet files using glob pattern
                files = glob.glob(data_path)
                if not files:
                    print(f"Warning: No files found matching pattern {data_path}")
                    continue
                # Load and concatenate all matching parquet files
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                self._process_dataframe(df, data_path)
            elif data_path.endswith(".csv"):
                # Load CSV file
                self._load_csv_data(data_path)
            else:
                # Get data list from JSON/JSONL
                with open(data_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in tqdm(lines, desc=f"Reading data list from {data_path}"):
                    if isinstance(line, str):
                        line = json.loads(line)
                    if not isinstance(line, dict):
                        continue

                    # Handle text/transcription columns - use 'text' as fallback
                    if "sentence" not in line and "sentences" not in line:
                        if "text" in line:
                            line["sentence"] = line["text"]
                        elif "transcription" in line:
                            line["sentence"] = line["transcription"]
                        elif "transcript" in line:
                            line["sentence"] = line["transcript"]

                    # Skip audio that exceeds duration limits
                    if line["duration"] < self.min_duration:
                        continue
                    if self.max_duration != -1 and line["duration"] > self.max_duration:
                        continue
                    # Skip audio that exceeds sentence character count limits
                    if "sentence" in line.keys():
                        if (
                            len(line["sentence"]) < self.min_sentence
                            or len(line["sentence"]) > self.max_sentence
                        ):
                            continue
                    elif "sentences" in line.keys():
                        sentence_len = 0
                        for s in line["sentences"]:
                            if isinstance(s, dict):
                                sentence_len += len(s.get("text", ""))
                            else:
                                sentence_len += len(str(s))
                        if (
                            sentence_len < self.min_sentence
                            or sentence_len > self.max_sentence
                        ):
                            continue
                    self.data_list.append(dict(line))

    def _process_dataframe(self, df, data_path):
        """Process a pandas DataFrame loaded from parquet file(s)."""
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing parquet data from {data_path}",
        ):
            # Convert row to dictionary
            line = row.to_dict()

            # Handle different possible column names and structures
            # Common parquet dataset formats may have different column names

            # Map common column names to expected format
            if "audio" in line and isinstance(line["audio"], dict):
                # Already in expected format
                pass
            elif (
                "audio_path" in line
                or "path" in line
                or "file" in line
                or "file_name" in line
            ):
                # Create audio dict from path column
                audio_path = (
                    line.get("audio_path")
                    or line.get("path")
                    or line.get("file")
                    or line.get("file_name")
                )
                line["audio"] = {"path": audio_path}
                if "start_time" in line:
                    line["audio"]["start_time"] = line["start_time"]
                if "end_time" in line:
                    line["audio"]["end_time"] = line["end_time"]

            # Handle text/transcription columns - prioritize 'text' column
            if "sentence" not in line and "sentences" not in line:
                # First try 'text' column as primary fallback
                if "text" in line:
                    line["sentence"] = line["text"]
                elif "transcription" in line:
                    line["sentence"] = line["transcription"]
                elif "transcript" in line:
                    line["sentence"] = line["transcript"]
            # Also handle case where 'sentence' exists but is empty/null
            elif "sentence" in line and (
                line["sentence"] is None or line["sentence"] == ""
            ):
                if "text" in line and line["text"] is not None and line["text"] != "":
                    line["sentence"] = line["text"]

            # Ensure duration field exists
            if "duration" not in line:
                # If duration is missing, try to infer from start/end times or set a default
                if (
                    "audio" in line
                    and "start_time" in line["audio"]
                    and "end_time" in line["audio"]
                ):
                    line["duration"] = (
                        line["audio"]["end_time"] - line["audio"]["start_time"]
                    )
                else:
                    # Skip if we can't determine duration
                    continue

            # Apply filtering criteria
            if line["duration"] < self.min_duration:
                continue
            if self.max_duration != -1 and line["duration"] > self.max_duration:
                continue

            # Check sentence length limits
            if "sentence" in line.keys():
                if (
                    len(line["sentence"]) < self.min_sentence
                    or len(line["sentence"]) > self.max_sentence
                ):
                    continue
            elif "sentences" in line.keys():
                sentence_len = 0
                for s in line["sentences"]:
                    if isinstance(s, dict):
                        sentence_len += len(s.get("text", ""))
                    else:
                        sentence_len += len(str(s))
                if sentence_len < self.min_sentence or sentence_len > self.max_sentence:
                    continue

            self.data_list.append(dict(line))

    def _load_csv_data(self, data_path):
        """Load data from CSV file. Supports multiple CSV formats."""
        with open(data_path, "r", encoding="utf-8") as f:
            # Try to detect CSV format by reading first few lines
            sample_lines = []
            for i, line in enumerate(f):
                sample_lines.append(line.strip())
                if i >= 2:  # Read first 3 lines to detect format
                    break

        # Reset file pointer
        with open(data_path, "r", encoding="utf-8") as f:
            # Detect delimiter and format
            has_header = False
            delimiter = ","

            # Check if first line looks like a header
            first_line = sample_lines[0] if sample_lines else ""
            if any(
                keyword in first_line.lower()
                for keyword in [
                    "filename",
                    "path",
                    "audio",
                    "text",
                    "sentence",
                    "transcript",
                ]
            ):
                has_header = True

            # Check for pipe delimiter (LJSpeech format)
            if "|" in first_line and "," not in first_line:
                delimiter = "|"

            reader = csv.reader(f, delimiter=delimiter)

            # Skip header if present
            if has_header:
                header = next(reader)
                print(f"Detected CSV header: {header}")

            for row in tqdm(reader, desc=f"Reading CSV data from {data_path}"):
                if len(row) < 2:
                    continue

                # Extract filename and text from row
                if delimiter == "|":
                    # LJSpeech format: filename|text
                    if "|" in row[0] and len(row) == 1:
                        filename, text = row[0].split("|", 1)
                    else:
                        filename, text = row[0], row[1] if len(row) > 1 else ""
                else:
                    # Standard CSV format: filename,text or audio_path,transcription
                    filename, text = row[0], row[1]

                # Create line dict in expected format
                line = {"audio": {"path": filename}, "sentence": text.strip()}

                # Try to get audio duration if file exists
                try:
                    if os.path.isfile(filename):
                        audio_path = filename
                    else:
                        # Try relative to CSV file directory
                        csv_dir = os.path.dirname(data_path)
                        audio_path = os.path.join(csv_dir, filename)

                    if os.path.isfile(audio_path):
                        line["audio"]["path"] = audio_path
                        sample, sr = soundfile.read(audio_path)
                        duration = round(sample.shape[-1] / float(sr), 2)
                        line["duration"] = duration
                    else:
                        # Skip if audio file not found
                        print(f"Warning: Audio file not found: {filename}")
                        continue

                except Exception as e:
                    print(f"Warning: Could not read audio file {filename}: {e}")
                    continue

                # Apply filtering criteria
                if "duration" in line:
                    if line["duration"] < self.min_duration:
                        continue
                    if self.max_duration != -1 and line["duration"] > self.max_duration:
                        continue

                # Check sentence length limits
                if (
                    len(line["sentence"]) < self.min_sentence
                    or len(line["sentence"]) > self.max_sentence
                ):
                    continue

                self.data_list.append(line)

    # Get audio data, sample rate, and text from data list
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # Split audio path and labels
        audio_file = data_list["audio"]["path"]

        # Handle transcript extraction with fallback to 'text' column
        if self.timestamps:
            transcript = data_list.get("sentences", data_list.get("text", ""))
        else:
            # Try 'sentence' first, then fall back to 'text'
            transcript = data_list.get("sentence", data_list.get("text", ""))

        language = data_list["language"] if "language" in data_list.keys() else None

        if "start_time" not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype="float32")
        else:
            start_time, end_time = (
                data_list["audio"]["start_time"],
                data_list["audio"]["end_time"],
            )
            # Split and read audio
            sample, sample_rate = self.slice_from_file(
                audio_file, start=start_time, end=end_time
            )
        sample = sample.T

        # Convert to mono channel
        if self.mono:
            sample = librosa.to_mono(sample)

        # ------------------------------
        # Example A: remove silence for non-timestamp training
        # (We avoid changing time alignment when timestamps=True)
        # ------------------------------
        if not self.timestamps:
            sample = remove_silence_librosa(
                sample,
                sample_rate,
                top_db=self.silence_top_db,
                min_silence_ms=self.silence_min_gap_ms,
                pad_ms=self.silence_pad_ms,
            )

        # Data augmentation (after silence removal)
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)

        # Resample - only downsample to 16000 when original sample rate is higher than 16000
        if sample_rate > 16000:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        elif self.sample_rate != sample_rate:
            sample = self.resample(
                sample, orig_sr=sample_rate, target_sr=self.sample_rate
            )

        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), (
            f"transcript should be list, current type: {type(transcript)}"
        )
        data = dict()
        labels = self.processor.tokenizer.get_vocab()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # Encode target text as label IDs
            start = (
                t["start"] if round(t["start"] * 100) % 2 == 0 else t["start"] + 0.01
            )
            if self.timestamp_begin is None:
                start = self.vocab[f"<|{start:.2f}|>"]
            else:
                start = self.timestamp_begin + round(start * 100) // 2
            end = t["end"] if round(t["end"] * 100) % 2 == 0 else t["end"] - 0.01
            if self.timestamp_begin is None:
                end = self.vocab[f"<|{end:.2f}|>"]
            else:
                end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t["text"]).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data["labels"] = labels + [self.endoftext]
        return data

    def __getitem__(self, idx):
        try:
            # Get audio data, sample rate, and text from data list
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # Can set language for individual data
            self.processor.tokenizer.set_prefix_tokens(
                language=language if language is not None else self.language
            )
            if len(transcript) > 0:
                # Load text with timestamps
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # Calculate log-Mel input features from input audio array
                    data["input_features"] = self.processor(
                        audio=sample, sampling_rate=self.sample_rate
                    ).input_features
                else:
                    # Get log-Mel features and label IDs
                    data = self.processor(
                        audio=sample, sampling_rate=self.sample_rate, text=transcript
                    )
            else:
                # If there's no text, use <|nospeech|> token
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data["labels"] = [self.startoftranscript, self.nospeech, self.endoftext]
            return data
        except Exception as e:
            print(
                f"Error reading data, index: {idx}, error message: {e}", file=sys.stderr
            )
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

    # Split and read audio
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # Count from the end
        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration
        # Ensure data doesn't go out of bounds
        if start < 0.0:
            start = 0.0
        if end > duration:
            end = duration
        if end < 0.0:
            raise ValueError("Slice end position (%f s) out of bounds" % end)
        if start > end:
            raise ValueError(
                "Slice start position (%f s) is later than slice end position (%f s)"
                % (start, end)
            )
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype="float32")
        return sample, sample_rate

    # Data augmentation
    def augment(self, sample, sample_rate):
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
                sample = self.change_speed(sample, speed_rate=rate)
            if config["type"] == "shift" and random.random() < config["prob"]:
                min_shift_ms, max_shift_ms = (
                    config["params"]["min_shift_ms"],
                    config["params"]["max_shift_ms"],
                )
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config["type"] == "volume" and random.random() < config["prob"]:
                min_gain_dBFS, max_gain_dBFS = (
                    config["params"]["min_gain_dBFS"],
                    config["params"]["max_gain_dBFS"],
                )
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config["type"] == "resample" and random.random() < config["prob"]:
                new_sample_rates = config["params"]["new_sample_rates"]
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(
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
                    sample = self.add_noise(
                        sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB
                    )
        return sample, sample_rate

    # Change speech speed
    @staticmethod
    def change_speed(sample, speed_rate):
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

    # Audio shift
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
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

    # Change volume
    @staticmethod
    def volume(sample, gain):
        sample *= 10.0 ** (gain / 20.0)
        return sample

    # Audio resampling
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # Add noise
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # Normalize audio volume to ensure noise is not too loud
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10.0 ** (gain / 20.0)
        # Specify noise volume
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10.0 ** (noise_gain_db / 20.0)
        # Fix noise length
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), "wrap")
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame : sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample**2)
        return 10 * np.log10(mean_square)
