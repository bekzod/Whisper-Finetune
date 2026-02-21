#!/usr/bin/env python3
"""Evaluate or transcribe with a NeMo Hybrid RNNT-CTC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a NeMo checkpoint (.ckpt/.nemo) on a manifest or a single audio file."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .ckpt/.nemo file, run directory, or checkpoints directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="NeMo JSONL manifest to evaluate (expects audio_filepath and text).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Single audio file to transcribe.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for transcription.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples from manifest (0 = all).",
    )
    parser.add_argument(
        "--decoder",
        choices=("rnnt", "ctc"),
        default="rnnt",
        help="Decoding branch to use (best-effort switch for hybrid models).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for inference.",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Apply local Uzbek normalization to refs/preds before WER.",
    )
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional output JSONL with audio/ref/pred for manifest mode.",
    )
    return parser.parse_args()


def find_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    search_dirs = [path]
    ckpt_subdir = path / "checkpoints"
    if ckpt_subdir.is_dir():
        search_dirs.insert(0, ckpt_subdir)

    candidates: List[Path] = []
    for root in search_dirs:
        candidates.extend(root.glob("*.nemo"))
        candidates.extend(root.glob("*.ckpt"))

    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt/.nemo found under {path} (or its checkpoints/ subdir)"
        )

    # Prefer explicit "last" checkpoints for restart/testing convenience.
    last_like = [c for c in candidates if "last" in c.name.lower()]
    pool = last_like if last_like else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def resolve_device(device_flag: str) -> str:
    import torch

    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path: Path, device: str):
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    map_location = torch.device(device)
    suffix = ckpt_path.suffix.lower()

    if suffix == ".nemo":
        model = EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=str(ckpt_path), map_location=map_location
        )
    elif suffix == ".ckpt":
        model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), map_location=map_location
        )
    else:
        raise ValueError(f"Unsupported checkpoint extension: {ckpt_path.suffix}")

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    return model


def set_decoder(model, decoder: str) -> None:
    if not hasattr(model, "change_decoding_strategy"):
        return
    try:
        model.change_decoding_strategy(decoder_type=decoder)
    except TypeError:
        # Some NeMo versions only support changing RNNT cfg here.
        if decoder == "rnnt":
            return
        print(
            "Warning: unable to switch to CTC decoder in this NeMo version; using default decoder."
        )
    except Exception as exc:  # pragma: no cover - runtime compatibility path
        print(f"Warning: failed to switch decoder to {decoder}: {exc}")


def safe_normalize(text: str, enabled: bool) -> str:
    if not enabled:
        return text.strip()
    try:
        from utils import normalize_text  # local file: nemo/utils.py

        return normalize_text(text)
    except Exception:
        return text.strip()


def to_text(prediction) -> str:
    if isinstance(prediction, str):
        return prediction
    if hasattr(prediction, "text"):
        return str(prediction.text)
    return str(prediction)


def transcribe_paths(model, paths: Sequence[str], batch_size: int) -> List[str]:
    try:
        outputs = model.transcribe(paths2audio_files=list(paths), batch_size=batch_size)
    except TypeError:
        outputs = model.transcribe(audio=list(paths), batch_size=batch_size)

    if isinstance(outputs, tuple) and outputs:
        # Some versions return (predictions, logits/metadata)
        outputs = outputs[0]
    return [to_text(item) for item in outputs]


def iter_manifest(manifest_path: Path, limit: int = 0) -> Iterable[Tuple[str, str]]:
    count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            audio = obj.get("audio_filepath")
            text = obj.get("text")
            if not audio or text is None:
                continue
            yield str(audio), str(text)
            count += 1
            if limit and count >= limit:
                break


def edit_distance(words_ref: Sequence[str], words_hyp: Sequence[str]) -> int:
    n = len(words_ref)
    m = len(words_hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        r = words_ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r == words_hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[m]


def evaluate_manifest(
    model,
    manifest_path: Path,
    batch_size: int,
    limit: int,
    normalize_text: bool,
    predictions_out: Path | None,
) -> None:
    pairs = list(iter_manifest(manifest_path, limit=limit))
    if not pairs:
        raise RuntimeError(f"No valid entries found in manifest: {manifest_path}")

    audio_paths = [p[0] for p in pairs]
    references = [safe_normalize(p[1], normalize_text) for p in pairs]

    total_edits = 0
    total_ref_words = 0
    processed = 0

    writer = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        writer = predictions_out.open("w", encoding="utf-8")

    try:
        for start in range(0, len(audio_paths), batch_size):
            end = min(start + batch_size, len(audio_paths))
            batch_audio = audio_paths[start:end]
            batch_ref = references[start:end]
            batch_pred = transcribe_paths(model, batch_audio, batch_size=batch_size)
            if len(batch_pred) != len(batch_audio):
                raise RuntimeError(
                    f"Prediction count mismatch for batch [{start}:{end}] "
                    f"(got {len(batch_pred)} expected {len(batch_audio)})"
                )

            for audio, ref, pred_raw in zip(batch_audio, batch_ref, batch_pred):
                pred = safe_normalize(pred_raw, normalize_text)
                ref_words = ref.split()
                hyp_words = pred.split()
                edits = edit_distance(ref_words, hyp_words)
                total_edits += edits
                total_ref_words += len(ref_words)
                processed += 1

                if writer is not None:
                    sample_wer = (edits / len(ref_words)) if ref_words else 0.0
                    writer.write(
                        json.dumps(
                            {
                                "audio_filepath": audio,
                                "ref": ref,
                                "pred": pred,
                                "wer": sample_wer,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if processed % max(batch_size * 5, 1) == 0:
                print(f"Processed {processed}/{len(audio_paths)} samples...")
    finally:
        if writer is not None:
            writer.close()

    if total_ref_words == 0:
        raise RuntimeError(
            "No reference words found in manifest text (cannot compute WER)."
        )

    wer = total_edits / total_ref_words
    print("\nEvaluation complete")
    print(f"Samples: {processed}")
    print(f"Words (ref): {total_ref_words}")
    print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
    if predictions_out is not None:
        print(f"Predictions written to: {predictions_out}")


def transcribe_single(model, audio_path: Path, normalize_text: bool) -> None:
    preds = transcribe_paths(model, [str(audio_path)], batch_size=1)
    if not preds:
        raise RuntimeError("No prediction returned for input audio.")
    text = safe_normalize(preds[0], normalize_text)
    print(text)


def main() -> None:
    args = parse_args()
    if not args.manifest and not args.audio:
        raise SystemExit("Provide either --manifest or --audio.")

    ckpt = find_checkpoint(Path(args.checkpoint))
    device = resolve_device(args.device)

    print(f"Using checkpoint: {ckpt}")
    print(f"Using device: {device}")
    model = load_model(ckpt, device=device)
    set_decoder(model, args.decoder)

    if args.audio:
        transcribe_single(
            model=model,
            audio_path=Path(args.audio).expanduser().resolve(),
            normalize_text=args.normalize_text,
        )
        return

    manifest_path = Path(args.manifest).expanduser().resolve()
    predictions_out = (
        Path(args.predictions_out).expanduser().resolve()
        if args.predictions_out
        else None
    )
    evaluate_manifest(
        model=model,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        limit=args.limit,
        normalize_text=args.normalize_text,
        predictions_out=predictions_out,
    )


if __name__ == "__main__":
    main()
