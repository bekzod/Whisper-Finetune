import os
import shutil
from typing import Any

import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import WhisperProcessor

from .data_utils import DataCollatorSpeechSeq2SeqWithPadding


# 保存模型时的回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank == 0 or args.local_rank == -1:
            # 保存效果最好的模型
            best_checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best"
            )
            # 因为只保存最新5个检查点，所以要确保不是之前的检查点
            if state.best_model_checkpoint is not None and os.path.exists(
                state.best_model_checkpoint
            ):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(
                f"效果最好的检查点为：{state.best_model_checkpoint}，评估结果为：{state.best_metric}"
            )
        return control


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
