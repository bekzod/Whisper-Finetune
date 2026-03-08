import os
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from nemo.utils import logging

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency
    HfApi = None

# Set WandB API key
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")
os.environ["WANDB_API_KEY"] = "2dfc22d8af7805df156e7f31ea3bc090ec99d52e"

torch.set_float32_matmul_precision("high")


class FreezeEncoderForSteps(pl.Callback):
    def __init__(self, freeze_steps: int):
        self.freeze_steps = max(0, int(freeze_steps))
        self._unfrozen = False

    @staticmethod
    def _set_encoder_trainable(pl_module, enabled: bool) -> None:
        if enabled:
            if hasattr(pl_module, "unfreeze_encoder"):
                pl_module.unfreeze_encoder()
                return
        else:
            if hasattr(pl_module, "freeze_encoder"):
                pl_module.freeze_encoder()
                return

        encoder = getattr(pl_module, "encoder", None)
        if encoder is None:
            return
        for param in encoder.parameters():
            param.requires_grad = enabled

    def on_fit_start(self, trainer, pl_module):
        if self.freeze_steps <= 0:
            return
        if trainer.global_step >= self.freeze_steps:
            self._unfrozen = True
            return
        self._set_encoder_trainable(pl_module, enabled=False)
        logging.info(
            f"Encoder frozen for first {self.freeze_steps} optimization steps."
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.freeze_steps <= 0 or self._unfrozen:
            return
        if trainer.global_step < self.freeze_steps:
            return
        self._set_encoder_trainable(pl_module, enabled=True)
        self._unfrozen = True
        logging.info(f"Encoder unfrozen at global_step={trainer.global_step}.")


class UploadBestToHFHub(pl.Callback):
    def __init__(
        self,
        repo_id: str,
        token_env: str = "HF_TOKEN",
        private: bool = False,
        path_in_repo: str = "checkpoints",
    ):
        self.repo_id = repo_id
        self.token_env = token_env
        self.private = bool(private)
        self.path_in_repo = (path_in_repo or "").strip("/")
        self._api = None
        self._token: Optional[str] = None
        self._last_uploaded_key: Optional[str] = None

    @staticmethod
    def _get_checkpoint_callback(trainer) -> Optional[ModelCheckpoint]:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback
        return None

    def _ensure_hf_client(self) -> bool:
        if self._api is not None:
            return True
        if HfApi is None:
            logging.error(
                "huggingface_hub is not installed. Install it or disable hf_hub.enabled."
            )
            return False

        token = os.environ.get(self.token_env)
        if not token:
            logging.error(
                f"{self.token_env} is not set. Skipping Hugging Face Hub uploads."
            )
            return False

        self._token = token
        self._api = HfApi()
        try:
            self._api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
                token=self._token,
            )
        except Exception as e:
            logging.error(f"Failed to create/access HF repo {self.repo_id}: {e}")
            self._api = None
            return False
        return True

    def _upload_best_model(self, trainer) -> None:
        checkpoint_callback = self._get_checkpoint_callback(trainer)
        if checkpoint_callback is None:
            return

        best_model_path = getattr(checkpoint_callback, "best_model_path", None)
        if not best_model_path:
            return

        best_path = Path(best_model_path)
        if not best_path.exists():
            return

        best_score = getattr(checkpoint_callback, "best_model_score", None)
        score_text = "unknown" if best_score is None else f"{float(best_score):.6f}"
        upload_key = f"{best_path.resolve()}::{score_text}"
        if upload_key == self._last_uploaded_key:
            return

        if not self._ensure_hf_client():
            return

        target_name = f"best_model{best_path.suffix}"
        path_in_repo = (
            f"{self.path_in_repo}/{target_name}" if self.path_in_repo else target_name
        )

        try:
            self._api.upload_file(
                path_or_fileobj=str(best_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                token=self._token,
                commit_message=(
                    f"Update best model at step={trainer.global_step}, score={score_text}"
                ),
            )
            self._last_uploaded_key = upload_key
            logging.info(
                f"Uploaded best model to hf://{self.repo_id}/{path_in_repo} "
                f"(score={score_text})."
            )
        except Exception as e:
            logging.warning(f"Failed to upload best model to Hugging Face Hub: {e}")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        self._upload_best_model(trainer)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._upload_best_model(trainer)


@hydra_runner(config_path=".", config_name="train")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # Debug: Print GPU info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    freeze_steps = int(cfg.get("freeze_encoder_steps", 0) or 0)
    if freeze_steps > 0:
        callbacks = list(trainer_cfg.get("callbacks") or [])
        callbacks.append(FreezeEncoderForSteps(freeze_steps=freeze_steps))
        trainer_cfg["callbacks"] = callbacks
        logging.info(f"Enabled encoder freezing for {freeze_steps} steps.")

    hf_cfg = cfg.get("hf_hub")
    if hf_cfg and bool(hf_cfg.get("enabled", False)):
        repo_id = hf_cfg.get("repo_id")
        if not repo_id:
            raise ValueError("hf_hub.repo_id must be set when hf_hub.enabled=true.")

        callbacks = list(trainer_cfg.get("callbacks") or [])
        callbacks.append(
            UploadBestToHFHub(
                repo_id=str(repo_id),
                token_env=str(hf_cfg.get("token_env", "HF_TOKEN")),
                private=bool(hf_cfg.get("private", False)),
                path_in_repo=str(hf_cfg.get("path_in_repo", "checkpoints")),
            )
        )
        trainer_cfg["callbacks"] = callbacks
        logging.info(f"Enabled Hugging Face Hub uploads to repo: {repo_id}")

    logging.info("Creating trainer...")
    trainer = pl.Trainer(**trainer_cfg)

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
