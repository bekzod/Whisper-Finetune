import os

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from nemo.utils import logging

# NeMo uses np.sctypes, which was removed in NumPy 2.x.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64, np.intp],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64, np.uintp],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.bytes_, np.str_, np.void],
    }

torch.set_float32_matmul_precision("medium")


def _get_spm_vocab_size(asr_model):
    try:
        return int(asr_model.tokenizer.tokenizer.get_piece_size())
    except Exception:
        return None


def _reapply_tdt_kwargs(asr_model, cfg):
    # Desired values from config
    desired_fastemit = float(cfg.model.loss.tdt_kwargs.fastemit_lambda)
    desired_clamp = float(cfg.model.loss.tdt_kwargs.clamp)

    # Update model cfg (authoritative for NeMo modules)
    try:
        asr_model._cfg.loss.tdt_kwargs.fastemit_lambda = desired_fastemit
        asr_model._cfg.loss.tdt_kwargs.clamp = desired_clamp
    except Exception as e:
        logging.warning(f"Could not set asr_model._cfg.loss.tdt_kwargs: {e}")

    # Try to update instantiated loss too
    try:
        if hasattr(asr_model, "loss") and hasattr(asr_model.loss, "tdt_kwargs"):
            if isinstance(asr_model.loss.tdt_kwargs, dict):
                asr_model.loss.tdt_kwargs["fastemit_lambda"] = desired_fastemit
                asr_model.loss.tdt_kwargs["clamp"] = desired_clamp
    except Exception as e:
        logging.warning(f"Could not update instantiated loss kwargs: {e}")

    # Best-effort rebuild loss module (version dependent)
    for fn_name in ("_init_loss", "_setup_loss", "setup_loss"):
        if hasattr(asr_model, fn_name):
            try:
                out = getattr(asr_model, fn_name)()
                if out is not None and fn_name == "_init_loss":
                    asr_model.loss = out
                break
            except Exception as e:
                logging.warning(f"Loss rebuild via {fn_name} failed: {e}")


def _ensure_target_tokenizer(asr_model, cfg, target_size=1024):
    expected_dir = str(cfg.model.tokenizer.dir)
    expected_type = str(cfg.model.tokenizer.type)

    vs = _get_spm_vocab_size(asr_model)
    logging.info(
        f"AFTER pretrained init: tokenizer vocab_size={vs}, target_dir={expected_dir}"
    )

    if vs is None:
        logging.warning(
            "Could not read tokenizer vocab size; cannot assert tokenizer consistency."
        )
        return

    if vs != target_size:
        logging.warning(
            f"Tokenizer vocab_size={vs} != {target_size}. Attempting to reset vocabulary."
        )
        if hasattr(asr_model, "change_vocabulary"):
            asr_model.change_vocabulary(
                new_tokenizer_dir=expected_dir,
                new_tokenizer_type=expected_type,
            )
            vs2 = _get_spm_vocab_size(asr_model)
            logging.info(f"Tokenizer reset attempted. New vocab_size={vs2}")
        else:
            raise RuntimeError(
                "Tokenizer mismatch but model has no change_vocabulary()."
            )


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # Debug GPU info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    logging.info("Creating trainer...")
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))

    logging.info("Setting up exp_manager...")
    exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info("Creating ASR model...")
    asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)

    logging.info(
        f"BEFORE pretrained init: tokenizer vocab_size={_get_spm_vocab_size(asr_model)}"
    )

    logging.info("Checking for pretrained checkpoint...")
    # Call with cfg.model if supported, else fallback to cfg
    try:
        asr_model.maybe_init_from_pretrained_checkpoint(cfg.model)
    except Exception:
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # Re-apply critical settings AFTER pretrained restore
    _reapply_tdt_kwargs(asr_model, cfg)
    _ensure_target_tokenizer(asr_model, cfg, target_size=1024)

    # Final sanity
    try:
        tdt = asr_model._cfg.loss.tdt_kwargs
        logging.info(
            f"AFTER pretrained init: fastemit_lambda={float(tdt.fastemit_lambda)}, clamp={float(tdt.clamp)}"
        )
    except Exception:
        logging.info(
            "AFTER pretrained init: could not read asr_model._cfg.loss.tdt_kwargs"
        )

    logging.info("Starting trainer.fit()...")
    trainer.fit(asr_model)

    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()
