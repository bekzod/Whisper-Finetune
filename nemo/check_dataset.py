import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_dataset

ds = load_dataset("bekzod123/uzbek_voice_3", split="train")
text_col = next(k for k in ds.column_names if k in ("text", "sentence", "transcription"))
print(f"Text column: {text_col}")
for i, item in enumerate(ds):
    print(f"[{i}] {item[text_col]}")
