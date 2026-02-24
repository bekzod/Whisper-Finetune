from datasets import load_dataset

ds = load_dataset("bekzod123/uzbek_voice_3", split="train", streaming=True)
item = next(iter(ds))
print("Columns:", list(item.keys()))
print("First item text keys:")
for k, v in item.items():
    if isinstance(v, str):
        print(f"  {k}: {v}")
