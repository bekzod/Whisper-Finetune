import re
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union

import torch
import numpy as np


# Remove punctuation marks
def remove_punctuation(text: str or List[str]):
    punctuation = "!,.;:?、！，。；：？"
    if isinstance(text, str):
        text = re.sub(r"[{}]+".format(punctuation), "", text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r"[{}]+".format(punctuation), "", t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"Unsupported type {type(text)}")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # e.g., WhisperProcessor
    label_pad_token_id: int = -100
    remove_bos_token: bool = True
    pad_to_multiple_of: Optional[int] = None  # e.g., 8/16 for better perf
    audio_key: str = "input_features"  # sometimes "input_values"
    labels_key: str = "labels"  # or "label_ids"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # ---- 1) AUDIO ----
        # Be resilient to datasets that wrap features as [tensor] vs tensor
        def _extract_audio(feat):
            x = feat[self.audio_key]
            # Many HF datasets store as list/array of length 1
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return x[0]
            return x

        input_features = [{self.audio_key: _extract_audio(f)} for f in features]

        # Return attention mask so the model can ignore padded frames
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", padding=True
        )
        # Some feature extractors use "attention_mask" for audio, otherwise add one if missing
        if "attention_mask" not in batch:
            # Create mask of 1s where frames exist (non-zero length along time axis).
            # We assume shape (B, C, T) or (B, T, F); safest is: mask by nonzero across feature dim.
            feats = batch[self.audio_key]
            # Reduce across non-time dims to detect non-padded frames
            # (handles both (B, T, F) and (B, C, T) layouts by summing abs values)
            if feats.dim() == 3:
                # Heuristic: treat last dim as feature dim when two spatial dims exist
                reduce_dim = -1
                amask = (feats.abs().sum(dim=reduce_dim) != 0).to(torch.long)
            else:
                # Fallback: ones mask
                amask = torch.ones(feats.shape[0], feats.shape[-1], dtype=torch.long)
            batch["attention_mask"] = amask

        # ---- 2) LABELS ----
        # Accept either `labels` or `label_ids` and normalize to a python list per sample
        def _extract_labels(feat):
            raw = feat.get(self.labels_key, feat.get("label_ids", None))
            if raw is None:
                raise KeyError(
                    f"Missing '{self.labels_key}' (or 'label_ids') in features."
                )
            # Convert to list[int]
            if isinstance(raw, torch.Tensor):
                return raw.tolist()
            return list(raw)

        label_seqs: List[List[int]] = [_extract_labels(f) for f in features]

        # Optionally strip a leading BOS on a PER-SEQUENCE basis BEFORE padding
        if (
            self.remove_bos_token
            and getattr(self.processor.tokenizer, "bos_token_id", None) is not None
        ):
            bos_id = self.processor.tokenizer.bos_token_id
            label_seqs = [
                seq[1:] if len(seq) > 0 and seq[0] == bos_id else seq
                for seq in label_seqs
            ]

        # Tokenizer pad to the longest in the batch (optionally to a multiple for speed)
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_seqs},
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Replace tokenizer pad with ignore index for CE loss
        labels = labels_batch["input_ids"]
        if "attention_mask" in labels_batch:
            labels = labels.masked_fill(
                labels_batch["attention_mask"].ne(1), self.label_pad_token_id
            )
        else:
            # Fallback: assume tokenizer pad token id
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is None:
                raise ValueError(
                    "Tokenizer has no pad_token_id and no labels attention_mask was returned."
                )
            labels = torch.where(
                labels == pad_id,
                torch.full_like(labels, self.label_pad_token_id),
                labels,
            )

        # Ensure correct dtype for CE loss
        if labels.dtype != torch.long:
            labels = labels.to(torch.long)

        batch["labels"] = labels
        return batch
