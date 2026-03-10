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
  - includes best-effort likely_bad_* columns that point to the most suspicious
    mismatch between the manifest text and the ASR hypothesis

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
  - Streams results to disk incrementally — only one batch + a top-K heap
    are held in memory at a time.
  - Supports --resume to continue from where a previous run left off.
  - This version uses robust, model-agnostic signals:
      * WER / CER between existing text and model hypothesis
      * normalized edit distances
      * text-length mismatch
      * speaking-rate mismatch (chars/sec)
  - It does NOT auto-replace labels.
  - Review top suspicious items manually.
"""

import argparse
import copy
import glob as _glob
import heapq
import json
import logging
import math
import os
import re
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, TextIO, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None
from huggingface_hub import hf_hub_download
from jiwer import cer, wer

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
LOGGER = logging.getLogger(__name__)

TOP_K = 500


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


def coerce_duration_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value <= 0:
        return None
    return value


def get_duration_seconds(audio_path: str) -> float:
    if sf is None:
        return float("nan")
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def count_manifest_lines(path: str) -> int:
    """Quick first pass to count non-empty lines for progress bars."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_manifest(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one row at a time from a JSONL manifest."""
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

            yield item


def batchify_iter(it: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
    """Collect items from an iterator into batches."""
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_text(
    text: str,
    lowercase: bool,
    remove_punctuation: bool,
) -> List[str]:
    normalized = normalize_text(
        text, lowercase=lowercase, remove_punctuation=remove_punctuation
    )
    return normalized.split() if normalized else []


def _hypothesis_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "text"):
        return str(output.text)
    return str(output)


def _coerce_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _expand_confidences(
    words: List[str],
    confidences: Any,
    lowercase: bool,
    remove_punctuation: bool,
) -> Optional[Tuple[List[str], List[Optional[float]]]]:
    if not isinstance(confidences, (list, tuple)) or len(confidences) != len(words):
        return None

    expanded_words = []
    expanded_confidences = []
    for word, confidence in zip(words, confidences):
        normalized_parts = tokenize_text(
            word,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        )
        if not normalized_parts:
            continue
        coerced_confidence = _coerce_confidence(confidence)
        for part in normalized_parts:
            expanded_words.append(part)
            expanded_confidences.append(coerced_confidence)

    return expanded_words, expanded_confidences


def _extract_word_confidences(
    output: Any,
    lowercase: bool,
    remove_punctuation: bool,
) -> Optional[List[Optional[float]]]:
    hyp_text = _hypothesis_text(output)
    hyp_words = tokenize_text(
        hyp_text,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
    )
    if not hyp_words:
        return None

    direct_conf = _expand_confidences(
        hyp_text.split(),
        getattr(output, "word_confidence", None),
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
    )
    if direct_conf is not None and direct_conf[0] == hyp_words:
        return direct_conf[1]

    timestep = getattr(output, "timestep", None)
    if timestep is not None:
        timestep_conf = _expand_confidences(
            hyp_text.split(),
            getattr(timestep, "word_confidence", None),
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        )
        if timestep_conf is not None and timestep_conf[0] == hyp_words:
            return timestep_conf[1]

        words = getattr(timestep, "words", None)
        if isinstance(words, list):
            parsed_words = []
            parsed_confidences = []
            for item in words:
                if isinstance(item, dict):
                    word_text = item.get("word") or item.get("text")
                    conf = item.get("confidence")
                    if conf is None:
                        conf = item.get("word_confidence")
                else:
                    word_text = getattr(item, "word", None) or getattr(
                        item, "text", None
                    )
                    conf = getattr(item, "confidence", None)
                    if conf is None:
                        conf = getattr(item, "word_confidence", None)

                if not word_text:
                    continue

                normalized_word = normalize_text(
                    word_text,
                    lowercase=lowercase,
                    remove_punctuation=remove_punctuation,
                )
                if not normalized_word:
                    continue

                normalized_parts = normalized_word.split()
                coerced_confidence = _coerce_confidence(conf)
                for part in normalized_parts:
                    parsed_words.append(part)
                    parsed_confidences.append(coerced_confidence)

            if parsed_words == hyp_words:
                return parsed_confidences

    return None


def transcribe_batch(
    model,
    audio_paths: List[str],
    lowercase: bool,
    remove_punctuation: bool,
) -> List[Dict[str, Any]]:
    """
    Works with NeMo ASR models whose transcribe() may return:
      - list[str]
      - list[Hypothesis/text-like objects]
    """
    with torch.inference_mode():
        try:
            outputs = model.transcribe(
                audio_paths,
                batch_size=len(audio_paths),
                return_hypotheses=True,
            )
        except TypeError:
            outputs = model.transcribe(audio_paths, batch_size=len(audio_paths))

    if isinstance(outputs, tuple) and outputs:
        outputs = outputs[0]

    hyps = []
    for out in outputs:
        hyps.append(
            {
                "text": _hypothesis_text(out),
                "word_confidences": _extract_word_confidences(
                    out,
                    lowercase=lowercase,
                    remove_punctuation=remove_punctuation,
                ),
            }
        )

    return hyps


def _best_confidence(
    confidences: Optional[List[Optional[float]]], start: int, end: int
) -> Optional[float]:
    if not confidences or start >= end:
        return None
    values = [c for c in confidences[start:end] if c is not None]
    if not values:
        return None
    return min(values)


def identify_likely_bad_span(
    ref_raw: str,
    hyp_raw: str,
    hyp_word_confidences: Optional[List[Optional[float]]],
    lowercase: bool,
    remove_punctuation: bool,
) -> Dict[str, Any]:
    ref_words = tokenize_text(
        ref_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )
    hyp_words = tokenize_text(
        hyp_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )

    default_result = {
        "likely_bad_text": "",
        "likely_bad_model_text": "",
        "likely_bad_operation": "",
        "likely_bad_word_confidence": None,
        "likely_bad_has_word_confidence": False,
    }

    if not ref_words and not hyp_words:
        return default_result

    candidates = []
    matcher = SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        ref_span = " ".join(ref_words[i1:i2])
        hyp_span = " ".join(hyp_words[j1:j2])
        confidence = _best_confidence(hyp_word_confidences, j1, j2)

        if tag == "replace" and (i2 - i1) == (j2 - j1) and (i2 - i1) > 0:
            for offset in range(i2 - i1):
                token_conf = _best_confidence(
                    hyp_word_confidences, j1 + offset, j1 + offset + 1
                )
                candidates.append(
                    {
                        "likely_bad_text": ref_words[i1 + offset],
                        "likely_bad_model_text": hyp_words[j1 + offset],
                        "likely_bad_operation": tag,
                        "likely_bad_word_confidence": token_conf,
                        "likely_bad_has_word_confidence": token_conf is not None,
                        "_rank": (
                            0 if token_conf is not None else 1,
                            token_conf if token_conf is not None else 1.0,
                            0,
                            i1 + offset,
                            j1 + offset,
                        ),
                    }
                )
            continue

        if tag == "delete" and ref_span:
            for offset, word in enumerate(ref_words[i1:i2]):
                candidates.append(
                    {
                        "likely_bad_text": word,
                        "likely_bad_model_text": "",
                        "likely_bad_operation": tag,
                        "likely_bad_word_confidence": None,
                        "likely_bad_has_word_confidence": False,
                        "_rank": (1, 1.0, 0, i1 + offset, j1),
                    }
                )
            continue

        if tag == "insert" and hyp_span:
            for offset, word in enumerate(hyp_words[j1:j2]):
                token_conf = _best_confidence(
                    hyp_word_confidences, j1 + offset, j1 + offset + 1
                )
                candidates.append(
                    {
                        "likely_bad_text": "",
                        "likely_bad_model_text": word,
                        "likely_bad_operation": tag,
                        "likely_bad_word_confidence": token_conf,
                        "likely_bad_has_word_confidence": token_conf is not None,
                        "_rank": (
                            0 if token_conf is not None else 1,
                            token_conf if token_conf is not None else 1.0,
                            1,
                            i1,
                            j1 + offset,
                        ),
                    }
                )
            continue

        candidates.append(
            {
                "likely_bad_text": ref_span,
                "likely_bad_model_text": hyp_span,
                "likely_bad_operation": tag,
                "likely_bad_word_confidence": confidence,
                "likely_bad_has_word_confidence": confidence is not None,
                "_rank": (
                    0 if confidence is not None else 1,
                    confidence if confidence is not None else 1.0,
                    0 if tag in {"replace", "delete"} else 1,
                    i1,
                    j1,
                ),
            }
        )

    if not candidates:
        return default_result

    best = min(candidates, key=lambda item: item["_rank"])
    best.pop("_rank", None)
    return best


def compute_row_scores(
    ref_raw: str,
    hyp_raw: str,
    duration: float,
    lowercase: bool,
    remove_punctuation: bool,
    hyp_word_confidences: Optional[List[Optional[float]]] = None,
) -> Dict[str, Any]:
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
    wer_s = clamp01(row_wer)
    cer_s = clamp01(row_cer)
    word_len_s = clamp01(word_len_mismatch / 1.0)
    char_len_s = clamp01(char_len_mismatch / 1.0)
    cps_s = clamp01(cps_mismatch / 0.7)

    suspicion = (
        0.45 * wer_s
        + 0.30 * cer_s
        + 0.15 * char_len_s
        + 0.05 * word_len_s
        + 0.05 * cps_s
    )

    if suspicion >= 0.85:
        bucket = "very_high"
    elif suspicion >= 0.65:
        bucket = "high"
    elif suspicion >= 0.40:
        bucket = "medium"
    elif suspicion >= 0.25:
        bucket = "low"
    else:
        bucket = "lowest"

    metrics = {
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
    metrics.update(
        identify_likely_bad_span(
            ref_raw=ref_raw,
            hyp_raw=hyp_raw,
            hyp_word_confidences=hyp_word_confidences,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        )
    )
    return metrics


def estimate_max_workers(model_size_gb: float = 1.5, headroom_gb: float = 2.0) -> int:
    """Estimate how many model replicas fit in free VRAM."""
    if not torch.cuda.is_available():
        return 1
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    usable = max(0, free_gb - headroom_gb)
    n = max(1, int(usable // model_size_gb))
    return n


def configure_torch_runtime(device: str) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            LOGGER.debug("Could not set float32 matmul precision to 'high'")


def _collect_done_audio_paths(paths: List[str]) -> set:
    """Read existing result files and collect audio_filepath values already processed."""
    done = set()
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done.add(rec["audio_filepath"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def merge_result_files(target_path: str, source_paths: List[str]) -> int:
    """Append unique rows from source files into the target JSONL."""
    merged_lines = []
    seen_audio_paths = _collect_done_audio_paths([target_path])
    for source_path in source_paths:
        if not os.path.exists(source_path):
            continue
        with open(source_path, "r", encoding="utf-8") as src_f:
            for line in src_f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                    audio_path = rec["audio_filepath"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if audio_path in seen_audio_paths:
                    continue
                seen_audio_paths.add(audio_path)
                merged_lines.append(stripped)

    if not merged_lines:
        return 0

    open_mode = "a" if os.path.exists(target_path) else "w"
    with open(target_path, open_mode, encoding="utf-8") as out_f:
        for line in merged_lines:
            out_f.write(line + "\n")
    return len(merged_lines)


def _resolve_audio_path(
    audio_path: str, manifest_dir: str, resolve_relative: bool
) -> str:
    if resolve_relative and not os.path.isabs(audio_path):
        return str(Path(manifest_dir) / audio_path)
    return audio_path


def _duration_sort_key(row: Dict[str, Any]) -> Tuple[int, float, str]:
    duration = coerce_duration_seconds(row.get("duration"))
    if duration is None:
        return 1, float("inf"), str(row.get("audio_filepath", ""))
    return 0, duration, str(row.get("audio_filepath", ""))


def arrange_rows_for_better_batching(
    rows: List[Dict[str, Any]], preserve_manifest_order: bool
) -> List[Dict[str, Any]]:
    if preserve_manifest_order:
        return rows
    return sorted(rows, key=_duration_sort_key)


def make_worker_chunks(
    rows: List[Dict[str, Any]], batch_size: int, num_workers: int
) -> List[List[Dict[str, Any]]]:
    workers: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, num_workers))]
    for batch_index, batch in enumerate(batchify_iter(iter(rows), batch_size)):
        workers[batch_index % len(workers)].extend(batch)
    return [chunk for chunk in workers if chunk]


def _resolve_model_path(model_name: str, model_filename: str = None) -> str:
    """Return a local .nemo path, downloading from HF if model_filename is set."""
    if model_filename:
        return hf_hub_download(
            repo_id=model_name, filename=model_filename, repo_type="model"
        )
    return model_name


def _load_asr_model(model_name: str, model_filename: str = None, device: str = "cuda"):
    """Load a NeMo ASR model from a local path, HF repo file, or pretrained name."""
    if nemo_asr is None:
        raise ImportError(
            "nemo_toolkit[asr] is required to run rank_bad_labels.py. "
            "Install it with `pip install nemo_toolkit[asr]`."
        )
    path = _resolve_model_path(model_name, model_filename)
    if path.endswith(".nemo") and os.path.exists(path):
        model = nemo_asr.models.ASRModel.restore_from(path, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=path)
    try:
        decoding_cfg = copy.deepcopy(model.cfg.decoding)
        if hasattr(decoding_cfg, "confidence_cfg"):
            confidence_cfg = decoding_cfg.confidence_cfg
            if hasattr(confidence_cfg, "preserve_frame_confidence"):
                confidence_cfg.preserve_frame_confidence = True
            if hasattr(confidence_cfg, "preserve_token_confidence"):
                confidence_cfg.preserve_token_confidence = True
            if hasattr(confidence_cfg, "preserve_word_confidence"):
                confidence_cfg.preserve_word_confidence = True
        model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    except Exception as exc:
        LOGGER.debug("Could not enable NeMo confidence outputs: %s", exc)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    return model


def resolve_row_audio_path(
    row: Dict[str, Any], manifest_dir: str, resolve_relative: bool
) -> None:
    row["audio_filepath"] = _resolve_audio_path(
        row["audio_filepath"], manifest_dir, resolve_relative
    )


def ensure_row_duration(row: Dict[str, Any]) -> None:
    duration = coerce_duration_seconds(row.get("duration"))
    if duration is not None:
        row["duration"] = duration
        return
    try:
        row["duration"] = get_duration_seconds(row["audio_filepath"])
    except Exception:
        row["duration"] = float("nan")


def _score_batch_outputs(
    batch_rows: List[Dict[str, Any]],
    hyps: List[Dict[str, Any]],
    lowercase: bool,
    remove_punctuation: bool,
) -> List[str]:
    if len(batch_rows) != len(hyps):
        raise RuntimeError(
            f"Expected {len(batch_rows)} hypotheses, received {len(hyps)}"
        )

    scored_lines = []
    for row, hyp_info in zip(batch_rows, hyps):
        ensure_row_duration(row)
        hyp = hyp_info["text"]
        metrics = compute_row_scores(
            ref_raw=row["text"],
            hyp_raw=hyp,
            duration=row.get("duration", float("nan")),
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            hyp_word_confidences=hyp_info.get("word_confidences"),
        )
        out = dict(row)
        out["model_hypothesis"] = hyp
        out.update(metrics)
        scored_lines.append(json.dumps(out, ensure_ascii=False))
    return scored_lines


def _write_scored_lines(out_f: TextIO, scored_lines: List[str]) -> int:
    for line in scored_lines:
        out_f.write(line + "\n")
    out_f.flush()
    return len(scored_lines)


def process_rows(
    model,
    rows: List[Dict[str, Any]],
    batch_size: int,
    manifest_dir: str,
    resolve_relative: bool,
    lowercase: bool,
    remove_punctuation: bool,
    out_f: TextIO,
    postprocess_workers: int,
    progress_desc: Optional[str] = None,
) -> int:
    total_batches = math.ceil(len(rows) / max(1, batch_size))
    batch_indices = range(0, len(rows), batch_size)
    if progress_desc is not None:
        batch_indices = tqdm(
            batch_indices,
            total=total_batches,
            desc=progress_desc,
        )

    executor = (
        ThreadPoolExecutor(max_workers=postprocess_workers)
        if postprocess_workers > 0
        else None
    )
    pending: Deque[Future[List[str]]] = deque()
    max_pending_batches = max(1, postprocess_workers * 2)
    rows_written = 0

    try:
        for i in batch_indices:
            batch_rows = rows[i : i + batch_size]
            for row in batch_rows:
                resolve_row_audio_path(row, manifest_dir, resolve_relative)

            audio_paths = [r["audio_filepath"] for r in batch_rows]
            hyps = transcribe_batch(
                model,
                audio_paths,
                lowercase=lowercase,
                remove_punctuation=remove_punctuation,
            )

            if executor is None:
                rows_written += _write_scored_lines(
                    out_f,
                    _score_batch_outputs(
                        batch_rows,
                        hyps,
                        lowercase=lowercase,
                        remove_punctuation=remove_punctuation,
                    ),
                )
                continue

            pending.append(
                executor.submit(
                    _score_batch_outputs,
                    batch_rows,
                    hyps,
                    lowercase,
                    remove_punctuation,
                )
            )
            if len(pending) >= max_pending_batches:
                rows_written += _write_scored_lines(out_f, pending.popleft().result())

        while pending:
            rows_written += _write_scored_lines(out_f, pending.popleft().result())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return rows_written


def _worker_transcribe(
    worker_id: int,
    model_name: str,
    model_filename: str,
    device: str,
    chunk: List[Dict[str, Any]],
    batch_size: int,
    manifest_dir: str,
    resolve_relative: bool,
    lowercase: bool,
    remove_punctuation: bool,
    output_path: str,
    postprocess_workers: int,
):
    """Worker process: loads its own model replica, transcribes its chunk, writes results."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | worker-{worker_id} | %(levelname)s | %(message)s",
    )
    log = logging.getLogger(f"worker-{worker_id}")

    if not chunk:
        log.info("Nothing to do, all rows already processed")
        return

    configure_torch_runtime(device)
    log.info("Loading model: %s (filename=%s)", model_name, model_filename)
    model = _load_asr_model(model_name, model_filename, device)
    log.info("Model ready, processing %d rows", len(chunk))

    with open(output_path, "a", encoding="utf-8") as out_f:
        written = process_rows(
            model=model,
            rows=chunk,
            batch_size=batch_size,
            manifest_dir=manifest_dir,
            resolve_relative=resolve_relative,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            out_f=out_f,
            postprocess_workers=postprocess_workers,
        )

    log.info("Done: wrote %d rows to %s", written, output_path)


def prepare_row(row: Dict[str, Any], manifest_dir: str, resolve_relative: bool) -> None:
    """Resolve paths and fill missing durations in-place."""
    resolve_row_audio_path(row, manifest_dir, resolve_relative)
    ensure_row_duration(row)


def _clean_text_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _most_common_value(values: pd.Series, default: str = "") -> str:
    counts = Counter()
    for value in values:
        cleaned = _clean_text_cell(value)
        if cleaned:
            counts[cleaned] += 1
    if not counts:
        return default
    return counts.most_common(1)[0][0]


def _mean_or_none(values: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def build_bad_word_report(df: pd.DataFrame) -> pd.DataFrame:
    report_columns = [
        "bad_text",
        "count",
        "top_model_text",
        "top_operation",
        "avg_suspicion_score",
        "avg_wer",
        "avg_bad_word_confidence",
        "example_text",
        "example_model_hypothesis",
    ]
    if df.empty:
        return pd.DataFrame(columns=report_columns)

    work_df = df.copy()
    work_df["bad_text"] = work_df["likely_bad_text"].map(_clean_text_cell)
    work_df["top_model_text"] = work_df["likely_bad_model_text"].map(_clean_text_cell)
    work_df["top_operation"] = work_df["likely_bad_operation"].map(_clean_text_cell)

    work_df = work_df[
        (work_df["bad_text"] != "") | (work_df["top_model_text"] != "")
    ].copy()
    if work_df.empty:
        return pd.DataFrame(columns=report_columns)

    work_df.loc[work_df["bad_text"] == "", "bad_text"] = "<missing_in_text>"
    work_df.loc[work_df["top_model_text"] == "", "top_model_text"] = (
        "<missing_in_model>"
    )
    work_df.loc[work_df["top_operation"] == "", "top_operation"] = "unknown"

    rows = []
    grouped = work_df.groupby("bad_text", sort=False, dropna=False)
    for bad_text, group in grouped:
        ranked_group = group.sort_values(
            ["suspicion_score", "wer", "cer"], ascending=False
        )
        top_row = ranked_group.iloc[0]
        rows.append(
            {
                "bad_text": bad_text,
                "count": int(len(group)),
                "top_model_text": _most_common_value(
                    group["top_model_text"], default="<missing_in_model>"
                ),
                "top_operation": _most_common_value(
                    group["top_operation"], default="unknown"
                ),
                "avg_suspicion_score": float(group["suspicion_score"].mean()),
                "avg_wer": float(group["wer"].mean()),
                "avg_bad_word_confidence": _mean_or_none(
                    group["likely_bad_word_confidence"]
                ),
                "example_text": _clean_text_cell(top_row.get("text", "")),
                "example_model_hypothesis": _clean_text_cell(
                    top_row.get("model_hypothesis", "")
                ),
            }
        )

    report = pd.DataFrame(rows, columns=report_columns)
    if report.empty:
        return report

    return report.sort_values(
        ["count", "avg_suspicion_score", "avg_wer"],
        ascending=False,
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="NeMo/HF model name or local .nemo path",
    )
    parser.add_argument(
        "--model-filename",
        default=None,
        help="Filename inside the HF repo to download (e.g. 'nemo_asr_2.nemo'). "
        "When set, uses hf_hub_download to fetch the file from --model repo.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from a previous interrupted run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel model replicas (0 = auto-detect from free VRAM)",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=None,
        help="CPU worker threads for duration probing + scoring. "
        "Default: 2 when using CUDA, otherwise 0.",
    )
    parser.add_argument(
        "--preserve-manifest-order",
        action="store_true",
        default=False,
        help="Disable duration-based reordering and keep the original manifest order.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.postprocess_workers is not None and args.postprocess_workers < 0:
        parser.error("--postprocess-workers must be >= 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    configure_torch_runtime(args.device)

    manifest_dir = str(Path(args.manifest).resolve().parent)
    unsorted_path = f"{args.output_prefix}._unsorted.jsonl"

    # Clean stale worker/unsorted files when NOT resuming
    if not args.resume:
        for stale in _glob.glob(f"{args.output_prefix}._worker_*.jsonl"):
            os.remove(stale)
            LOGGER.info("Removed stale worker file: %s", stale)
        if os.path.exists(unsorted_path):
            os.remove(unsorted_path)
            LOGGER.info("Removed stale unsorted file: %s", unsorted_path)

    # Count total rows for progress bar
    LOGGER.info("Counting manifest rows: %s", args.manifest)
    total_rows = count_manifest_lines(args.manifest)
    LOGGER.info("Total rows: %d", total_rows)

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers <= 0:
        if args.device == "cuda" and torch.cuda.is_available():
            num_workers = estimate_max_workers(model_size_gb=1.5, headroom_gb=2.0)
        else:
            num_workers = 1
    LOGGER.info("Using %d parallel worker(s)", num_workers)

    postprocess_workers = args.postprocess_workers
    if postprocess_workers is None:
        postprocess_workers = 2 if args.device == "cuda" else 0
    LOGGER.info("Using %d post-processing worker thread(s)", postprocess_workers)

    existing_worker_paths = sorted(_glob.glob(f"{args.output_prefix}._worker_*.jsonl"))
    resume_paths = [unsorted_path] + existing_worker_paths
    done_audio_paths: set = set()
    if args.resume:
        done_audio_paths = _collect_done_audio_paths(resume_paths)
        if done_audio_paths:
            LOGGER.info(
                "Resume: found %d already-processed rows in prior outputs",
                len(done_audio_paths),
            )

    # Read all remaining rows into memory for chunking
    remaining_rows = []
    skipped_rows = 0
    for row in iter_manifest(args.manifest):
        resolve_row_audio_path(row, manifest_dir, args.resolve_relative_to_manifest)
        if done_audio_paths and row["audio_filepath"] in done_audio_paths:
            skipped_rows += 1
            continue
        remaining_rows.append(row)

    remaining_rows = arrange_rows_for_better_batching(
        remaining_rows,
        preserve_manifest_order=args.preserve_manifest_order,
    )
    remaining = len(remaining_rows)
    LOGGER.info("Rows to process: %d (skipped %d)", remaining, skipped_rows)

    if num_workers > 1 and remaining > 0:
        # --- Multi-worker parallel path ---
        mp.set_start_method("spawn", force=True)

        chunks = make_worker_chunks(remaining_rows, args.batch_size, num_workers)
        actual_workers = len(chunks)

        # Stable worker output files (survive interrupts for resume)
        worker_paths = [
            f"{args.output_prefix}._worker_{i}.jsonl" for i in range(actual_workers)
        ]

        LOGGER.info(
            "Spawning %d workers (%d batch-sized chunks distributed round-robin)",
            actual_workers,
            math.ceil(remaining / max(1, args.batch_size)),
        )

        processes = []
        for i, (chunk, wpath) in enumerate(zip(chunks, worker_paths)):
            p = mp.Process(
                target=_worker_transcribe,
                args=(
                    i,
                    args.model,
                    args.model_filename,
                    args.device,
                    chunk,
                    args.batch_size,
                    manifest_dir,
                    args.resolve_relative_to_manifest,
                    args.lowercase,
                    args.remove_punctuation,
                    wpath,
                    postprocess_workers,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check for failures
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                LOGGER.error("Worker %d exited with code %d", i, p.exitcode)

    elif remaining > 0:
        # --- Single-worker path (original logic) ---
        LOGGER.info("Loading model: %s (filename=%s)", args.model, args.model_filename)
        model = _load_asr_model(args.model, args.model_filename, args.device)
        LOGGER.info("Model ready on device: %s", args.device)

        open_mode = "a" if (args.resume and done_audio_paths) else "w"

        with open(unsorted_path, open_mode, encoding="utf-8") as out_f:
            process_rows(
                model=model,
                rows=remaining_rows,
                batch_size=args.batch_size,
                manifest_dir=manifest_dir,
                resolve_relative=args.resolve_relative_to_manifest,
                lowercase=args.lowercase,
                remove_punctuation=args.remove_punctuation,
                out_f=out_f,
                postprocess_workers=postprocess_workers,
                progress_desc="Transcribing & scoring",
            )

    merged_rows = merge_result_files(
        unsorted_path,
        sorted(
            set(existing_worker_paths)
            | set(_glob.glob(f"{args.output_prefix}._worker_*.jsonl"))
        ),
    )
    if merged_rows:
        LOGGER.info("Merged %d worker-shard rows into %s", merged_rows, unsorted_path)

    LOGGER.info(
        "Transcription pass complete: %d rows processed",
        remaining + skipped_rows,
    )

    # Build bucket_counts and top_k_heap from the full unsorted file
    top_k_heap: List[Tuple[float, int, Dict]] = []
    bucket_counts: Counter = Counter()
    rows_processed = 0
    if os.path.exists(unsorted_path):
        with open(unsorted_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                bucket_counts[rec["bucket"]] += 1
                score = rec["suspicion_score"]
                entry = (score, idx, rec)
                if len(top_k_heap) < TOP_K:
                    heapq.heappush(top_k_heap, entry)
                elif score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, entry)
                rows_processed += 1

    # Sort and write final outputs
    LOGGER.info("Reading unsorted results for final sort")
    df = pd.read_json(unsorted_path, lines=True)
    df = df.sort_values(["suspicion_score", "wer", "cer"], ascending=False).reset_index(
        drop=True
    )

    csv_path = f"{args.output_prefix}.csv"
    jsonl_path = f"{args.output_prefix}.jsonl"
    top500_path = f"{args.output_prefix}.top500.csv"
    bad_words_path = f"{args.output_prefix}.bad_words.csv"

    LOGGER.info("Writing outputs with prefix: %s", args.output_prefix)
    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Top 500 from heap (already in memory, no need to read from disk)
    top_k_sorted = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
    top_k_rows = [entry[2] for entry in top_k_sorted]
    pd.DataFrame(top_k_rows).to_csv(top500_path, index=False)
    bad_words_df = build_bad_word_report(df)
    bad_words_df.to_csv(bad_words_path, index=False)

    LOGGER.info("Saved: %s", csv_path)
    LOGGER.info("Saved: %s", jsonl_path)
    LOGGER.info("Saved: %s", top500_path)
    LOGGER.info("Saved: %s", bad_words_path)

    # Clean up temp files
    try:
        os.remove(unsorted_path)
    except OSError:
        pass
    # Remove worker shard files from multi-worker runs
    for wf in _glob.glob(f"{args.output_prefix}._worker_*.jsonl"):
        try:
            os.remove(wf)
        except OSError:
            pass

    # Summary
    print()
    print("Bucket counts:")
    for bucket in ["very_high", "high", "medium", "low"]:
        print(f"  {bucket:>10s}  {bucket_counts.get(bucket, 0)}")

    print()
    print("Top 10 suspicious rows:")
    cols = [
        "audio_filepath",
        "duration",
        "suspicion_score",
        "wer",
        "cer",
        "likely_bad_text",
        "likely_bad_model_text",
        "likely_bad_word_confidence",
        "text",
        "model_hypothesis",
    ]
    top10 = [entry[2] for entry in top_k_sorted[:10]]
    print(pd.DataFrame(top10)[cols].to_string(index=False))

    print()
    print("Top 20 likely bad words:")
    word_cols = [
        "bad_text",
        "count",
        "top_model_text",
        "top_operation",
        "avg_suspicion_score",
        "avg_bad_word_confidence",
    ]
    if bad_words_df.empty:
        print("  (none)")
    else:
        print(bad_words_df.head(20)[word_cols].to_string(index=False))


if __name__ == "__main__":
    main()
