import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import WhisperProcessor

from .data_utils import DataCollatorSpeechSeq2SeqWithPadding


# Callback invoked when saving the model
class SavePeftModelCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self._last_recorded_best: Optional[str] = None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank not in (0, -1):
            return control

        best_checkpoint = state.best_model_checkpoint
        if not best_checkpoint or not os.path.exists(best_checkpoint):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        # unwrap DeepSpeed/Accelerate containers
        peft_model = getattr(model, "module", model)
        if not isinstance(peft_model, PeftModel):
            # Avoid copying multi-gigabyte full fine-tune checkpoints; just log best path.
            self._record_best_path(args.output_dir, best_checkpoint)
            return control

        if self._last_recorded_best == best_checkpoint:
            return control

        best_checkpoint_folder = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-best"
        try:
            if best_checkpoint_folder.exists():
                shutil.rmtree(best_checkpoint_folder)
            best_checkpoint_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            warnings.warn(f"Failed to refresh best-checkpoint folder: {exc}")
            return control

        try:
            peft_model.save_pretrained(
                best_checkpoint_folder, safe_serialization=True
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to export PEFT adapters for best checkpoint: {exc}")
            return control

        # Preserve trainer metadata and create a pointer back to the full checkpoint.
        trainer_state_src = Path(best_checkpoint) / "trainer_state.json"
        if trainer_state_src.exists():
            try:
                shutil.copy2(
                    trainer_state_src, best_checkpoint_folder / "trainer_state.json"
                )
            except OSError as exc:
                warnings.warn(f"Failed to copy trainer_state.json: {exc}")

        self._record_best_path(args.output_dir, best_checkpoint)
        self._last_recorded_best = best_checkpoint
        print(
            f"Best checkpoint: {state.best_model_checkpoint}, eval metric: {state.best_metric}"
        )
        return control

    def _record_best_path(
        self,
        output_dir: str,
        checkpoint_path: str,
    ) -> None:
        """Persist the latest best checkpoint path for tooling."""
        pointer_file = Path(output_dir) / "best_checkpoint_path.txt"
        try:
            pointer_file.write_text(f"{checkpoint_path}\n")
        except OSError as exc:
            warnings.warn(f"Failed to record best checkpoint path: {exc}")


class WandbPredictionLogger(TrainerCallback):
    """Logs a small table of (pred, ref) on each eval_end."""

    def __init__(
        self,
        processor: WhisperProcessor,
        collator: DataCollatorSpeechSeq2SeqWithPadding,
        eval_dataset,
        max_rows: int = 6,
        use_wandb: bool = True,
    ):
        self.processor = processor
        self.collator = collator
        self.eval_dataset = eval_dataset
        self.max_rows = max_rows
        self.use_wandb = use_wandb

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.use_wandb:
            return
        try:
            import wandb
        except Exception:
            return

        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device

        # sample a few rows deterministically (first N)
        indices = list(range(min(self.max_rows, len(self.eval_dataset))))
        batch = [self.eval_dataset[i] for i in indices]
        batch = self.collator(batch)
        # move tensors to device
        batch_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k != "labels"
        }

        # ensure eval uses language/task prompt
        eval_forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=args.language, task=args.task
        )
        prev_forced = model.config.forced_decoder_ids
        model.config.forced_decoder_ids = eval_forced_decoder_ids

        with torch.no_grad():
            gen_ids = model.generate(
                **batch_on_device,
                max_length=args.generation_max_length,
            )

        # restore forced ids
        model.config.forced_decoder_ids = prev_forced

        # decode preds
        pred_str = self.processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        # decode refs
        labels = batch["labels"].clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        ref_str = self.processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        table = wandb.Table(columns=["idx", "prediction", "reference"])
        for i, (p, r) in enumerate(zip(pred_str, ref_str)):
            table.add_data(indices[i], p, r)

        wandb.log({"eval/predictions_table": table}, step=state.global_step)
