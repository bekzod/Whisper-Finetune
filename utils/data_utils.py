import re
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

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
    processor: Any
    pad_to_multiple_of: Optional[int] = 8  # good for tensor cores (bf16)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) audio features -> batch pad with feature_extractor
        # Accept either precomputed "input_features" or raw "audio" dicts
        if "input_features" in features[0]:
            input_features = [f["input_features"] for f in features]
            # Avoid slow torch.tensor(list_of_ndarrays); use NumPy stacking or torch.stack directly
            first = input_features[0]
            if torch.is_tensor(first):
                batch = {"input_features": torch.stack(input_features, dim=0)}
                # infer full-ones attention mask since features are precomputed/padded
                B, _, T = batch["input_features"].shape
                batch["attention_mask"] = torch.ones((B, T), dtype=torch.long)
            else:
                try:
                    if hasattr(first, "shape"):
                        # numpy-like arrays
                        np_batch = np.stack(input_features, axis=0).astype(
                            np.float32, copy=False
                        )
                    else:
                        # lists/tuples of floats
                        np_batch = np.asarray(input_features, dtype=np.float32)
                        if np_batch.dtype == object:
                            raise ValueError(
                                "Ragged input_features detected (dtype=object). Ensure equal shapes or provide raw 'audio' for vectorized padding."
                            )
                    batch = {"input_features": torch.from_numpy(np_batch)}
                    B, _, T = batch["input_features"].shape
                    batch["attention_mask"] = torch.ones((B, T), dtype=torch.long)
                except Exception as e:
                    raise ValueError(
                        f"Failed to batch input_features; ensure all feature arrays share the same shape. Original error: {e}"
                    )
        else:
            # Expect something like {"audio": {"array": np.ndarray, "sampling_rate": 16000}}
            audio_inputs = [f["audio"] for f in features]
            fe = self.processor.feature_extractor(
                [a["array"] for a in audio_inputs],
                sampling_rate=audio_inputs[0]["sampling_rate"],
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,  # batched pad (vectorized)
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            batch = {
                "input_features": fe.input_features,
                "attention_mask": fe.attention_mask,
            }

        # 2) labels -> prefer raw text if available; else fall back to ids
        if "text" in features[0]:
            texts = [f["text"] for f in features]
            tok = self.processor.tokenizer(  # __call__ fast path
                texts,
                padding=True,  # vectorized padding
                truncation=True,
                max_length=int(
                    min(
                        448,
                        getattr(self.processor.tokenizer, "model_max_length", 448)
                        or 448,
                    )
                ),
                return_tensors="pt",
            )
            labels = tok.input_ids
        else:
            # fallback if dataset already stored token ids (slower, but supported)
            label_features = [{"input_ids": f["labels"]} for f in features]
            tok = self.processor.tokenizer.pad(
                label_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            labels = tok["input_ids"]

        # HF expects -100 for ignored positions in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # Optional: decoder prompt for multilingual tasks (kept lean)
        # Trainer/compute_metrics already manages forced_decoder_ids; nothing here.

        return batch
