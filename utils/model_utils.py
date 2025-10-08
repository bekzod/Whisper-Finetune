import bitsandbytes as bnb
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def find_all_linear_names(use_8bit, model):
    cls = bnb.nn.Linear8bitLt if use_8bit else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_modules = list(lora_module_names)
    return target_modules


def load_from_checkpoint(resume_from_checkpoint, model=None):
    """
    Robust checkpoint loader that tolerates naming differences between OpenAI Whisper
    checkpoints (proj_out.weight) and HF transformers (lm_head.weight). It also
    handles sharded and safetensors checkpoints, and removes common prefixes like
    'module.' when resuming from DDP training.

    Expected to be assigned to Trainer._load_from_checkpoint and called with:
        - resume_from_checkpoint: str path to a checkpoint directory or file
        - model: the instantiated model to load into
    """
    # Only handle explicit string paths with a provided model
    if (
        not resume_from_checkpoint
        or not isinstance(resume_from_checkpoint, str)
        or model is None
    ):
        return

    import os
    import json
    import warnings
    from typing import Dict

    def _load_file(fp: str):
        if not os.path.exists(fp):
            return None
        if fp.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as st_load

                return st_load(fp, device="cpu")
            except Exception as e:
                warnings.warn(f"Failed to load safetensors file {fp}: {e}")
                return None
        try:
            return torch.load(fp, map_location="cpu")
        except Exception as e:
            warnings.warn(f"Failed to load torch file {fp}: {e}")
            return None

    ckpt_path = resume_from_checkpoint
    candidates: list[str] = []

    # If a directory was provided, probe common filenames (including sharded indices)
    if os.path.isdir(ckpt_path):
        # Common single-file checkpoints
        for fname in (
            "pytorch_model.bin",
            "pytorch_model.safetensors",
            "model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors",
        ):
            fpath = os.path.join(ckpt_path, fname)
            if os.path.exists(fpath):
                candidates.append(fpath)

        # Sharded checkpoints via index files
        for idx_name in (
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ):
            idx_path = os.path.join(ckpt_path, idx_name)
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        idx_data = json.load(f)
                    weight_map = idx_data.get("weight_map", {}) or {}
                    shard_files = sorted(set(weight_map.values()))
                    for shard in shard_files:
                        shard_path = os.path.join(ckpt_path, shard)
                        if os.path.exists(shard_path):
                            candidates.append(shard_path)
                except Exception as e:
                    warnings.warn(f"Failed to parse index file {idx_path}: {e}")
    else:
        # A direct file path was provided
        candidates.append(ckpt_path)

    if not candidates:
        warnings.warn(f"No checkpoint files found under {ckpt_path}")
        return

    # Load and merge state dicts (for sharded checkpoints)
    merged_state: Dict[str, torch.Tensor] = {}
    for fp in candidates:
        sd = _load_file(fp)
        if sd is None:
            continue
        # Some formats store actual tensors under 'state_dict'
        if (
            isinstance(sd, dict)
            and "state_dict" in sd
            and isinstance(sd["state_dict"], dict)
        ):
            sd = sd["state_dict"]
        # Merge
        for k, v in sd.items():
            merged_state[k] = v

    if not merged_state:
        warnings.warn(f"Checkpoint at {ckpt_path} contained no loadable tensors.")
        return

    # Key normalization and remapping:
    # - Strip common prefixes (e.g., DDP "module.")
    # - Map various proj_out/proj variants to lm_head.weight (with shape check)
    remapped: Dict[str, torch.Tensor] = {}
    did_map_proj_out = False
    did_skip_proj_out = False

    # Determine target lm_head key and expected shape from the current model
    model_sd = model.state_dict()
    target_lm_key = None
    if "lm_head.weight" in model_sd:
        target_lm_key = "lm_head.weight"
    elif "model.lm_head.weight" in model_sd:
        target_lm_key = "model.lm_head.weight"
    target_lm_shape = model_sd[target_lm_key].shape if target_lm_key else None

    for key, tensor in merged_state.items():
        new_key = key

        # Strip DDP prefix
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        # Some saved checkpoints may (rarely) prefix with "model." even for top-level heads
        if new_key.startswith("model.lm_head."):
            new_key = new_key[len("model.") :]

        # Map OpenAI/HF naming differences for the decoder output projection
        is_proj_variant = (
            new_key == "proj_out.weight"
            or new_key.endswith(".proj_out.weight")
            or new_key.endswith(".decoder.proj_out.weight")
            or new_key == "decoder.proj.weight"
            or new_key.endswith(".decoder.proj.weight")
        )
        if is_proj_variant:
            # Only map if shapes match to avoid loading wrong vocab sizes
            if target_lm_key and (tensor.shape == target_lm_shape):
                new_key = target_lm_key
                did_map_proj_out = True
            else:
                did_skip_proj_out = True
                # Skip adding this tensor to avoid unexpected-key warnings
                continue

        remapped[new_key] = tensor

    if did_map_proj_out:
        print(
            "Remapped decoder projection to 'lm_head.weight' (from proj_out/proj variants)."
        )
    if did_skip_proj_out:
        print(
            "Skipped mapping decoder projection due to shape mismatch; keeping model head weights."
        )

    # Finally, load with strict=False to tolerate non-critical extras/missing keys
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Improve messaging: filter out benign keys from warnings if needed
    if missing:
        # When mapping worked, 'proj_out.weight' should not be in missing keys anymore
        print(f"Checkpoint load: missing keys count={len(missing)}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys count={len(unexpected)}")
