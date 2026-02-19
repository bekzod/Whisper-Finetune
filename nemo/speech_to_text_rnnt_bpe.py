import os

# CUDA/Numba compatibility fixes for newer GPUs (Blackwell/RTX 50-series)
# os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
# os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"

# Fix CUDA memory fragmentation (helps avoid OOM on variable-length sequences)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set cache directories to avoid permission issues
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("NUMBA_CACHE_DIR", "/workspace/.cache/numba")

# Set WandB API key
os.environ["WANDB_API_KEY"] = "2dfc22d8af7805df156e7f31ea3bc090ec99d52e"

import lightning.pytorch as pl
import numpy as np
import torch

# NeMo uses np.sctypes, which was removed in NumPy 2.x.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64, np.intp],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64, np.uintp],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.bytes_, np.str_, np.void],
    }

# Optimize for Tensor Cores on supported GPUs (trades precision for performance)
torch.set_float32_matmul_precision("medium")
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from nemo.utils import logging


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

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
