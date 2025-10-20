#!/usr/bin/env python3
"""
Custom logits processors used to adjust Whisper decoding.

Currently includes:
* KenLMBiasLogitsProcessor - lightweight KenLM rescoring for top-k tokens.
* VocabularyFirstTokenBiasLogitsProcessor - boosts the first token for bias phrases.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import LogitsProcessor


class KenLMBiasLogitsProcessor(LogitsProcessor):
    """
    Rescales the logits for the top-k candidate tokens using an n-gram KenLM model.

    The processor scores each candidate by concatenating it to the current prefix,
    computing kenlm.score(new_text) - kenlm.score(prefix), and adding `alpha * delta`
    back into the logits. This provides a light-weight language model fusion step.
    """

    def __init__(
        self,
        kenlm_model: Any,
        tokenizer: Any,
        alpha: float = 0.5,
        top_k: int = 50,
    ) -> None:
        self.lm = kenlm_model
        self.tokenizer = tokenizer
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self._cache: Dict[str, float] = {}

    def _lm_score(self, text: str) -> float:
        if text in self._cache:
            return self._cache[text]
        score = self.lm.score(text.strip(), bos=False, eos=False)
        self._cache[text] = score
        return score

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        prefixes = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        vocab_size = scores.shape[-1]
        for batch_index, prefix in enumerate(prefixes):
            base_score = self._lm_score(prefix)
            k = min(self.top_k, vocab_size)
            topk = torch.topk(scores[batch_index], k)
            idxs = topk.indices
            deltas: List[float] = []
            for token_id in idxs.tolist():
                token_piece = self.tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                if not token_piece:
                    deltas.append(0.0)
                    continue
                new_text = prefix + token_piece
                delta = self._lm_score(new_text) - base_score
                deltas.append(self.alpha * float(delta))
            if deltas:
                delta_tensor = torch.tensor(
                    deltas, device=scores.device, dtype=scores.dtype
                )
                scores[batch_index, idxs] += delta_tensor
        return scores


class VocabularyFirstTokenBiasLogitsProcessor(LogitsProcessor):
    """
    Boosts the first token of each provided bias word/phrase by adding `alpha`.

    Implementation notes:
    * All bias phrases are pre-tokenized (with and without leading space) so we only
      need to store their first token id.
    * During `__call__` we add a cached bias mask to all logits in the batch.
    """

    def __init__(
        self, tokenizer: Any, bias_words: Iterable[str], alpha: float = 5.0
    ) -> None:
        self.tokenizer = tokenizer
        self.alpha = float(alpha)
        self._first_token_ids = self._build_first_token_set(bias_words)
        # Cache the bias mask per (vocab_size, device, dtype) tuple for efficiency.
        self._cached_masks: Dict[
            Tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}

    def _encode_without_special(self, text: str) -> Sequence[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            tokens = self.tokenizer(text, add_special_tokens=False)
            return tokens.get("input_ids", [])

    def _build_first_token_set(self, bias_words: Iterable[str]) -> List[int]:
        first_tokens: List[int] = []
        for word in bias_words:
            if not word:
                continue
            sequences = set()
            try:
                normal_ids = self._encode_without_special(word)
            except Exception:
                normal_ids = []
            if normal_ids:
                sequences.add(tuple(int(i) for i in normal_ids))
            try:
                spaced_ids = self._encode_without_special(" " + word)
            except Exception:
                spaced_ids = []
            if spaced_ids:
                sequences.add(tuple(int(i) for i in spaced_ids))
            for seq in sequences:
                if seq:
                    first_tokens.append(int(seq[0]))
        # Ensure deterministic ordering and remove duplicates
        return sorted(set(first_tokens))

    def _bias_mask(
        self, vocab_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cache_key = (vocab_size, device, dtype)
        mask = self._cached_masks.get(cache_key)
        if mask is None:
            mask = torch.zeros(vocab_size, device=device, dtype=dtype)
            for token_id in self._first_token_ids:
                if 0 <= token_id < vocab_size:
                    mask[token_id] += self.alpha
            self._cached_masks[cache_key] = mask
        return mask

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not self._first_token_ids:
            return scores
        vocab_size = scores.shape[-1]
        device = scores.device
        dtype = scores.dtype
        mask = self._bias_mask(vocab_size, device, dtype)
        scores += mask.unsqueeze(0)
        return scores
