"""
Convert a local Hugging Face Whisper checkpoint directory into an
OpenAI-format Whisper .pt file that whisper.load_model() accepts.

Usage:
  python convert_hf_whisper_to_openai.py --model_dir checkpoint-final --out whisper_converted.pt
"""

import argparse
import torch
from transformers import WhisperForConditionalGeneration

# ---------- Key renaming (HF -> OpenAI) ----------
# We map *specific* HF substrings to OpenAI substrings. Order matters:
# longest / most specific first to avoid partial collisions.
_HF_TO_OPENAI_PAIRS = [
    # Module stacks
    ("encoder.layers", "encoder.blocks"),
    ("decoder.layers", "decoder.blocks"),
    # Attention (self)
    (".self_attn.q_proj", ".attn.query"),
    (".self_attn.k_proj", ".attn.key"),
    (".self_attn.v_proj", ".attn.value"),
    (".self_attn.out_proj", ".attn.out"),
    (".self_attn_layer_norm", ".attn_ln"),
    # Attention (cross)
    (".encoder_attn.q_proj", ".cross_attn.query"),
    (".encoder_attn.k_proj", ".cross_attn.key"),
    (".encoder_attn.v_proj", ".cross_attn.value"),
    (".encoder_attn.out_proj", ".cross_attn.out"),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    # MLP
    (".fc1", ".mlp.0"),
    (".fc2", ".mlp.2"),
    (".final_layer_norm", ".mlp_ln"),
    # Block-level LNs on encoder/decoder stacks
    (
        "encoder.layer_norm.",
        "encoder.ln_post.",
    ),  # IMPORTANT: OpenAI expects ln_post on encoder
    ("decoder.layer_norm.", "decoder.ln."),
    # Positional & token embeddings
    ("encoder.embed_positions.weight", "encoder.positional_embedding"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
    ("embed_tokens", "token_embedding"),
    # Rare top-level layer_norm (not typical, but handle if present)
    ("layer_norm.", "ln_post."),
]


def _rename_key_hf_to_openai(key: str) -> str:
    # strip "model." prefix used by WhisperForConditionalGeneration
    if key.startswith("model."):
        key = key[len("model.") :]
    # Apply ordered replacements
    for hf_sub, oa_sub in _HF_TO_OPENAI_PAIRS:
        if hf_sub in key:
            key = key.replace(hf_sub, oa_sub)
    return key


def _build_dims_from_config(cfg) -> dict:
    # Match OpenAI Whisper's expected 'dims' payload
    return {
        "n_mels": cfg.num_mel_bins,
        "n_audio_ctx": cfg.max_source_positions,
        "n_audio_state": cfg.d_model,
        "n_audio_head": cfg.encoder_attention_heads,
        "n_audio_layer": cfg.encoder_layers,
        "n_vocab": cfg.vocab_size,
        "n_text_ctx": cfg.max_target_positions,
        "n_text_state": cfg.d_model,
        "n_text_head": cfg.decoder_attention_heads,
        "n_text_layer": cfg.decoder_layers,
    }


def _drop_hf_specific_heads(sd: dict):
    # HF uses a separate LM head; OpenAI ties it to token_embedding.
    for k in list(sd.keys()):
        if k.startswith("lm_head.") or k.startswith("proj_out."):
            sd.pop(k)


def _apply_encoder_ln_post_safety_patch(openai_sd: dict):
    """
    Ensure the encoder's final LN is named 'encoder.ln_post.*' as expected by OpenAI Whisper.
    If any 'encoder.ln.' slipped through (from older/inconsistent mappings), rename it.
    """
    # Bulk move if a prefix exists
    moved = 0
    for k in list(openai_sd.keys()):
        if k.startswith("encoder.ln."):
            new_k = "encoder.ln_post." + k[len("encoder.ln.") :]
            openai_sd[new_k] = openai_sd.pop(k)
            moved += 1

    # Minimal fallback if only weight/bias exist
    if "encoder.ln_post.weight" not in openai_sd and "encoder.ln.weight" in openai_sd:
        openai_sd["encoder.ln_post.weight"] = openai_sd.pop("encoder.ln.weight")
        moved += 1
    if "encoder.ln_post.bias" not in openai_sd and "encoder.ln.bias" in openai_sd:
        openai_sd["encoder.ln_post.bias"] = openai_sd.pop("encoder.ln.bias")
        moved += 1
    return moved


def _cast_state_dict_to_bfloat16(sd: dict) -> None:
    """Convert all floating-point tensors in-place to bfloat16 to shrink the checkpoint."""
    for key, tensor in list(sd.items()):
        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
            sd[key] = tensor.to(dtype=torch.bfloat16)


def convert(model_dir: str, out_path: str):
    # Load local HF checkpoint (reads .safetensors/.bin shards seamlessly)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=None,  # keep original dtype
        low_cpu_mem_usage=True,  # reduce peak RAM for large checkpoints
    )

    sd = model.state_dict()
    _drop_hf_specific_heads(sd)

    # Rename to OpenAI layout
    openai_sd = {}
    for k, v in sd.items():
        new_k = _rename_key_hf_to_openai(k)
        openai_sd[new_k] = v

    # Safety patch: guarantee encoder.ln_post.* exists (and not encoder.ln.*)
    _apply_encoder_ln_post_safety_patch(openai_sd)

    # Quick sanity checks
    required = [
        "decoder.token_embedding.weight",
        "decoder.positional_embedding",
        "encoder.positional_embedding",
        "encoder.ln_post.weight",
        "encoder.ln_post.bias",
    ]
    missing = [k for k in required if k not in openai_sd]
    if missing:
        raise RuntimeError(f"Missing required keys after rename: {missing}")

    dims = _build_dims_from_config(model.config)

    _cast_state_dict_to_bfloat16(openai_sd)

    torch.save({"dims": dims, "model_state_dict": openai_sd}, out_path)
    print(f"Saved OpenAI-format Whisper checkpoint to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="Path to HF checkpoint dir (has config.json, *.safetensors)",
    )
    ap.add_argument("--out", default="whisper_converted.pt", help="Output .pt filepath")
    args = ap.parse_args()
    convert(args.model_dir, args.out)


if __name__ == "__main__":
    main()
