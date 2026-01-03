from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from nemo.utils import logging


def _get_original_cwd() -> Path:
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def _split_comma_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _as_iterable_paths(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_comma_paths(value)
    if isinstance(value, (list, tuple)):
        paths: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                paths.extend(_split_comma_paths(item))
            else:
                paths.append(str(item))
        return paths
    return [str(value)]


def _to_abs_path(path: str, base_dir: Path) -> str:
    path_obj = Path(os.path.expanduser(path))
    if path_obj.is_absolute():
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def _abspath_in_cfg(cfg: Any, key_path: str, base_dir: Path) -> None:
    value = OmegaConf.select(cfg, key_path, default=None)
    if value is None:
        return

    if isinstance(value, str):
        OmegaConf.update(cfg, key_path, _to_abs_path(value, base_dir), merge=False)
        return

    if isinstance(value, (list, tuple)):
        OmegaConf.update(
            cfg,
            key_path,
            [_to_abs_path(str(p), base_dir) for p in _as_iterable_paths(value)],
            merge=False,
        )
        return

    OmegaConf.update(cfg, key_path, _to_abs_path(str(value), base_dir), merge=False)


def _require_existing_paths(paths: Iterable[str], what: str) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        missing_preview = "\n".join(f"  - {p}" for p in missing[:10])
        raise FileNotFoundError(
            f"Missing {what} path(s):\n{missing_preview}\n\n"
            "If you are using Hydra, relative paths may break because Hydra changes the working directory.\n"
            "Fix by either:\n"
            "  - running with `hydra.job.chdir=false`, or\n"
            "  - using absolute paths, or\n"
            "  - ensuring the files exist relative to where you launch the script."
        )


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    original_cwd = _get_original_cwd()
    logging.info(f"Process cwd: {Path.cwd()}")
    logging.info(f"Hydra original cwd: {original_cwd}")

    # Make common path fields absolute so they resolve correctly even if Hydra changes directories.
    try:
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass

    _abspath_in_cfg(cfg, "model.train_ds.manifest_filepath", original_cwd)
    _abspath_in_cfg(cfg, "model.validation_ds.manifest_filepath", original_cwd)
    _abspath_in_cfg(cfg, "model.test_ds.manifest_filepath", original_cwd)
    _abspath_in_cfg(cfg, "model.tokenizer.dir", original_cwd)
    _abspath_in_cfg(cfg, "model.train_ds.tarred_audio_filepaths", original_cwd)
    _abspath_in_cfg(cfg, "model.validation_ds.tarred_audio_filepaths", original_cwd)
    _abspath_in_cfg(cfg, "model.test_ds.tarred_audio_filepaths", original_cwd)

    # Fail fast with a clear error if datasets/tokenizer paths are missing.
    _require_existing_paths(
        _as_iterable_paths(OmegaConf.select(cfg, "model.train_ds.manifest_filepath")),
        "train manifest",
    )
    _require_existing_paths(
        _as_iterable_paths(
            OmegaConf.select(cfg, "model.validation_ds.manifest_filepath")
        ),
        "validation manifest",
    )
    tokenizer_dir = OmegaConf.select(cfg, "model.tokenizer.dir", default=None)
    if isinstance(tokenizer_dir, str):
        _require_existing_paths([tokenizer_dir], "tokenizer dir")

    # Debug: Print GPU info
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

    # Initialize the weights of the model from another model, if provided via config
    logging.info("Checking for pretrained checkpoint...")
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    logging.info("Starting trainer.fit()...")
    try:
        trainer.fit(asr_model)
    except Exception as e:
        logging.error(f"Training failed with exception: {e}")
        import traceback

        traceback.print_exc()
        raise

    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
