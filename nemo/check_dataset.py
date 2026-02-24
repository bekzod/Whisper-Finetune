import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_dataset

TEXT_CANDIDATES = (
    "text",
    "sentence",
    "transcription",
    "transcript",
    "label",
    "sentences",
)


def detect_text_column(ds):
    columns = list(ds.column_names)
    lookup = {str(name).lower(): name for name in columns}
    for key in TEXT_CANDIDATES:
        if key in lookup:
            return lookup[key]

    # Fallback: inspect a sample row for any known text-like keys.
    if len(ds) > 0:
        sample = ds[0]
        sample_lookup = {str(name).lower(): name for name in sample.keys()}
        for key in TEXT_CANDIDATES:
            if key in sample_lookup:
                return sample_lookup[key]

    raise ValueError(
        "Could not find transcript column. "
        f"Available columns: {columns}. Tried: {list(TEXT_CANDIDATES)}"
    )


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")).strip())
            else:
                parts.append(str(item).strip())
        return " ".join(p for p in parts if p)
    return str(value)


ds = load_dataset("bekzod123/uzbek_voice_3", split="train")
text_col = detect_text_column(ds)
print(f"Text column: {text_col}")
for i, item in enumerate(ds):
    print(f"[{i}] {to_text(item.get(text_col))}")
