import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_dataset

TEXT_CANDIDATES = (
    "text",
    "text_with_timestamp",
    "sentence",
    "transcription",
    "transcript",
    "sentences",
)


def detect_text_column(ds):
    columns = list(ds.column_names)
    lookup = {str(name).lower(): name for name in columns}

    def _sample_rows(max_rows=64):
        n = min(len(ds), max_rows)
        return [ds[i] for i in range(n)]

    def _looks_like_text(value):
        if value is None:
            return False
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return False
            return any(ch.isalpha() for ch in v)
        if isinstance(value, list):
            return len(to_text(value).strip()) > 0
        return False

    sample_rows = _sample_rows()

    # Prefer known transcript keys, but only if sampled values are text-like.
    for key in TEXT_CANDIDATES:
        col = lookup.get(key)
        if col is None:
            continue
        values = [row.get(col) for row in sample_rows]
        text_like = sum(1 for v in values if _looks_like_text(v))
        if text_like > 0:
            return col

    # Fallback: pick the most text-like column from all available columns.
    best_col = None
    best_score = -1
    for col in columns:
        values = [row.get(col) for row in sample_rows]
        score = sum(1 for v in values if _looks_like_text(v))
        if score > best_score:
            best_score = score
            best_col = col
    if best_col is not None and best_score > 0:
        return best_col

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
print(f"Columns: {ds.column_names}")
text_col = detect_text_column(ds)
print(f"Text column: {text_col}")
for i, item in enumerate(ds):
    print(f"[{i}] {to_text(item.get(text_col))}")
