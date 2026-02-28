#!/usr/bin/env python3
"""
rank_bad_labels.py

Rank likely-bad ASR labels by retranscribing audio with a NeMo / Parakeet model
and comparing the hypothesis to the existing transcript.

Input:
  JSONL manifest with at least:
    {"audio_filepath": "/abs/or/rel/path.wav", "text": "reference transcript"}
  Optional:
    {"duration": 12.34, "id": "utt_001"}

Output:
  - ranked CSV
  - ranked JSONL
  sorted by suspicion_score descending

Example:
  python rank_bad_labels.py \
    --manifest train_manifest.jsonl \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --batch-size 8 \
    --output-prefix ranked_train \
    --device cuda

Install:
  pip install nemo_toolkit['asr'] jiwer soundfile librosa pandas tqdm

Notes:
  - This version uses robust, model-agnostic signals:
      * WER / CER between existing text and model hypothesis
      * normalized edit distances
      * text-length mismatch
      * speaking-rate mismatch (chars/sec)
  - It does NOT auto-replace labels.
  - Review top suspicious items manually.
"""

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

# NeMo
import nemo.collections.asr as nemo_asr
from jiwer import cer, wer

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
LOGGER = logging.getLogger(__name__)


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def get_duration_seconds(audio_path: str) -> float:
    if sf is None:
        return float("nan")
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def load_manifest(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {lineno}: {e}") from e

            if "audio_filepath" not in item:
                raise ValueError(f"Line {lineno} missing 'audio_filepath'")
            if "text" not in item:
                raise ValueError(f"Line {lineno} missing 'text'")

            rows.append(item)
    return rows


def batchify(xs: List[Any], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def transcribe_batch(model, audio_paths: List[str]) -> List[str]:
    """
    Works with NeMo ASR models whose transcribe() may return:
      - list[str]
      - list[Hypothesis/text-like objects]
    """
    outputs = model.transcribe(audio_paths, batch_size=len(audio_paths))
    hyps = []

    for out in outputs:
        if isinstance(out, str):
            hyps.append(out)
        elif hasattr(out, "text"):
            hyps.append(out.text)
        else:
            hyps.append(str(out))

    return hyps


def compute_row_scores(
    ref_raw: str,
    hyp_raw: str,
    duration: float,
    lowercase: bool,
    remove_punctuation: bool,
) -> Dict[str, float]:
    ref = normalize_text(
        ref_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )
    hyp = normalize_text(
        hyp_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )

    ref_words = ref.split()
    hyp_words = hyp.split()

    ref_chars = len(ref.replace(" ", ""))
    hyp_chars = len(hyp.replace(" ", ""))

    # Distance signals
    row_wer = wer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)
    row_cer = cer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)

    # Length mismatch
    word_count_ratio = safe_div(len(hyp_words), max(1, len(ref_words)), default=1.0)
    char_count_ratio = safe_div(hyp_chars, max(1, ref_chars), default=1.0)

    # Symmetric mismatch around 1.0
    word_len_mismatch = abs(math.log(max(word_count_ratio, 1e-6)))
    char_len_mismatch = abs(math.log(max(char_count_ratio, 1e-6)))

    # Speaking-rate mismatch: chars/sec
    if duration and not math.isnan(duration) and duration > 0:
        ref_cps = ref_chars / duration
        hyp_cps = hyp_chars / duration
        cps_ratio = safe_div(hyp_cps, max(ref_cps, 1e-6), default=1.0)
        cps_mismatch = abs(math.log(max(cps_ratio, 1e-6)))
    else:
        ref_cps = float("nan")
        hyp_cps = float("nan")
        cps_mismatch = 0.0

    # Heuristic score: emphasize transcript disagreement first.
    # Saturate each component into [0,1] so one metric cannot dominate too hard.
    wer_s = clamp01(row_wer)  # already often in [0, 1+]
    cer_s = clamp01(row_cer)
    word_len_s = clamp01(word_len_mismatch / 1.0)  # ln(2) ~ 0.69 => noticeable mismatch
    char_len_s = clamp01(char_len_mismatch / 1.0)
    cps_s = clamp01(cps_mismatch / 0.7)

    suspicion = (
        0.45 * wer_s
        + 0.30 * cer_s
        + 0.15 * char_len_s
        + 0.05 * word_len_s
        + 0.05 * cps_s
    )

    # Buckets that are useful when listening
    if suspicion >= 0.85:
        bucket = "very_high"
    elif suspicion >= 0.65:
        bucket = "high"
    elif suspicion >= 0.40:
        bucket = "medium"
    else:
        bucket = "low"

    return {
        "ref_norm": ref,
        "hyp_norm": hyp,
        "wer": row_wer,
        "cer": row_cer,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "word_count_ratio": word_count_ratio,
        "char_count_ratio": char_count_ratio,
        "ref_chars_per_sec": ref_cps,
        "hyp_chars_per_sec": hyp_cps,
        "suspicion_score": suspicion,
        "bucket": bucket,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="NeMo/HF model name or local .nemo path",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-prefix", default="ranked_manifest")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--remove-punctuation", action="store_true", default=False)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--resolve-relative-to-manifest", action="store_true", default=False
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Loading manifest: %s", args.manifest)
    rows = load_manifest(args.manifest)
    LOGGER.info("Loaded %d rows", len(rows))
    manifest_dir = str(Path(args.manifest).resolve().parent)

    missing_duration = 0
    for row in tqdm(rows, desc="Preparing rows"):
        if args.resolve_relative_to_manifest and not os.path.isabs(
            row["audio_filepath"]
        ):
            row["audio_filepath"] = str(Path(manifest_dir) / row["audio_filepath"])

        if "duration" not in row or row["duration"] in (None, ""):
            missing_duration += 1
            try:
                row["duration"] = get_duration_seconds(row["audio_filepath"])
            except Exception:
                row["duration"] = float("nan")
    LOGGER.info("Rows missing duration: %d", missing_duration)

    # Load model
    LOGGER.info("Loading model: %s", args.model)
    if args.model.endswith(".nemo") and os.path.exists(args.model):
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

    if args.device == "cuda":
        model = model.cuda()
    LOGGER.info("Model ready on device: %s", args.device)

    # Transcribe
    audio_paths = [r["audio_filepath"] for r in rows]
    hypotheses: List[str] = []
    total_batches = math.ceil(len(audio_paths) / max(1, args.batch_size))
    LOGGER.info(
        "Starting transcription for %d files (%d batches, batch_size=%d)",
        len(audio_paths),
        total_batches,
        args.batch_size,
    )
    for batch in tqdm(
        batchify(audio_paths, args.batch_size), total=total_batches, desc="Transcribing"
    ):
        hyps = transcribe_batch(model, batch)
        hypotheses.extend(hyps)
    LOGGER.info("Transcription complete: %d hypotheses", len(hypotheses))

    if len(hypotheses) != len(rows):
        raise RuntimeError(
            f"Mismatch: got {len(hypotheses)} hypotheses for {len(rows)} rows"
        )

    # Score
    LOGGER.info("Scoring hypotheses against references")
    out_rows = []
    for row, hyp in tqdm(zip(rows, hypotheses), total=len(rows), desc="Scoring"):
        ref = row["text"]
        duration = row.get("duration", float("nan"))

        metrics = compute_row_scores(
            ref_raw=ref,
            hyp_raw=hyp,
            duration=duration,
            lowercase=args.lowercase,
            remove_punctuation=args.remove_punctuation,
        )

        out = dict(row)
        out["model_hypothesis"] = hyp
        out.update(metrics)
        out_rows.append(out)

    df = pd.DataFrame(out_rows)
    df = df.sort_values(["suspicion_score", "wer", "cer"], ascending=False).reset_index(
        drop=True
    )

    csv_path = f"{args.output_prefix}.csv"
    jsonl_path = f"{args.output_prefix}.jsonl"
    top500_path = f"{args.output_prefix}.top500.csv"

    LOGGER.info("Writing outputs with prefix: %s", args.output_prefix)
    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    df.head(500).to_csv(top500_path, index=False)

    LOGGER.info("Saved: %s", csv_path)
    LOGGER.info("Saved: %s", jsonl_path)
    LOGGER.info("Saved: %s", top500_path)

    # Simple summary
    print()
    print("Bucket counts:")
    print(df["bucket"].value_counts(dropna=False).to_string())

    print()
    print("Top 10 suspicious rows:")
    cols = [
        "audio_filepath",
        "duration",
        "suspicion_score",
        "wer",
        "cer",
        "text",
        "model_hypothesis",
    ]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
