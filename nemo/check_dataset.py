from datasets import load_dataset

ds = load_dataset(
    "bekzod123/uzbek_voice_3",
    data_files={"train": "train/metadata.csv"},  # adjust path
    split="train",
)
print(ds)
print(ds.column_names)
