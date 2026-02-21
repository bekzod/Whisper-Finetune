import os

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from nemo.utils import logging

# Set WandB API key
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")
os.environ["WANDB_API_KEY"] = "2dfc22d8af7805df156e7f31ea3bc090ec99d52e"

torch.set_float32_matmul_precision("medium")


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
