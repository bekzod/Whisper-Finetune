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
# Suppress Numba's low-occupancy warning spam (informational, not fatal).
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")

# Set WandB API key
os.environ["WANDB_API_KEY"] = "2dfc22d8af7805df156e7f31ea3bc090ec99d52e"

torch.set_float32_matmul_precision("medium")


def _get_spm_vocab_size(asr_model) -> int | None:
    """Best-effort tokenizer vocab size extraction (SentencePiece)."""
    try:
        tok = asr_model.tokenizer
        return int(tok.tokenizer.get_piece_size())
    except Exception:
        return None


def _reapply_tdt_kwargs(asr_model, cfg) -> None:
    """
    Re-apply TDT kwargs AFTER pretrained init.
    Some init flows can mutate cfg/loss objects.
    """
    # Desired values from Hydra cfg
    desired_fastemit = float(cfg.model.loss.tdt_kwargs.fastemit_lambda)
    desired_clamp = float(cfg.model.loss.tdt_kwargs.clamp)

    # Update model cfg
    try:
        asr_model._cfg.loss.tdt_kwargs.fastemit_lambda = desired_fastemit
        asr_model._cfg.loss.tdt_kwargs.clamp = desired_clamp
    except Exception as e:
        logging.warning(f"Could not set asr_model._cfg.loss.tdt_kwargs: {e}")

    # Update instantiated loss module if it exposes these fields
    try:
        if hasattr(asr_model, "loss") and hasattr(asr_model.loss, "tdt_kwargs"):
            # Some implementations store kwargs dict-like
            if isinstance(asr_model.loss.tdt_kwargs, dict):
                asr_model.loss.tdt_kwargs["fastemit_lambda"] = desired_fastemit
                asr_model.loss.tdt_kwargs["clamp"] = desired_clamp
            else:
                # Or attribute-style
                asr_model.loss.tdt_kwargs.fastemit_lambda = desired_fastemit
                asr_model.loss.tdt_kwargs.clamp = desired_clamp
    except Exception as e:
        logging.warning(f"Could not update instantiated loss tdt_kwargs: {e}")

    # Rebuild loss if NeMo exposes helpers (version-dependent)
    # This is the most reliable way to ensure the running graph uses the new kwargs.
    rebuilt = False
    for fn_name in ("_init_loss", "_setup_loss", "setup_loss"):
        if hasattr(asr_model, fn_name):
            try:
                fn = getattr(asr_model, fn_name)
                out = fn()
                # Some versions return loss, some set it internally
                if out is not None and fn_name == "_init_loss":
                    asr_model.loss = out
                rebuilt = True
                break
            except Exception as e:
                logging.warning(f"Loss rebuild via {fn_name}() failed: {e}")
    if not rebuilt:
        logging.info(
            "No loss rebuild helper found; relying on in-place loss kwargs update."
        )


def _ensure_target_tokenizer(asr_model, cfg) -> None:
    """
    Ensure the training model keeps the target tokenizer (e.g., 1024) after pretrained init.
    If mismatch is detected, attempt to reset vocabulary if the model supports it.
    """
    expected_dir = str(cfg.model.tokenizer.dir)
    expected_type = str(cfg.model.tokenizer.type)

    vocab_size = _get_spm_vocab_size(asr_model)
    logging.info(
        f"AFTER pretrained init: tokenizer vocab_size={vocab_size} (target dir={expected_dir})"
    )

    # If we can’t read vocab size, we still proceed (some tokenizer wrappers differ)
    if vocab_size is None:
        logging.warning(
            "Could not read tokenizer vocab size; ensure tokenizer is not being overwritten."
        )
        return

    # If you *know* it should be 1024, enforce it here:
    target_size = 1024
    if vocab_size != target_size:
        logging.warning(
            f"Tokenizer vocab_size={vocab_size} but expected {target_size}. "
            "Attempting to reset tokenizer/vocabulary to target."
        )

        # NeMo ASR BPE models typically provide change_vocabulary()
        if hasattr(asr_model, "change_vocabulary"):
            try:
                asr_model.change_vocabulary(
                    new_tokenizer_dir=expected_dir,
                    new_tokenizer_type=expected_type,
                )
                vocab_size2 = _get_spm_vocab_size(asr_model)
                logging.info(f"Tokenizer reset attempted. New vocab_size={vocab_size2}")
            except Exception as e:
                raise RuntimeError(
                    f"Tokenizer mismatch and change_vocabulary() failed: {e}"
                ) from e
        else:
            raise RuntimeError(
                "Tokenizer mismatch detected but model has no change_vocabulary() method. "
                "You must prevent pretrained init from overwriting the training model."
            )


@hydra_runner(config_path=".", config_name="train")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # Debug: GPU info
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

    # Sanity before init
    pre_vocab = _get_spm_vocab_size(asr_model)
    logging.info(f"BEFORE pretrained init: tokenizer vocab_size={pre_vocab}")

    # Initialize weights from pretrained model, if configured
    logging.info("Checking for pretrained checkpoint...")
    # Prefer passing model subtree if your NeMo version supports it; if it errors, revert to cfg
    try:
        asr_model.maybe_init_from_pretrained_checkpoint(cfg.model)
    except Exception:
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # Re-apply critical knobs AFTER init to prevent leakage from pretrained restore
    _reapply_tdt_kwargs(asr_model, cfg)
    _ensure_target_tokenizer(asr_model, cfg)

    # Print final TDT kwargs sanity (best-effort)
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
    try:
        trainer.fit(asr_model)
    except Exception as e:
        logging.error(f"Training failed with exception: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Optional test
    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa
