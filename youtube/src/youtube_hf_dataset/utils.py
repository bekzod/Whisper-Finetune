from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def require_binaries(names: Iterable[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required binaries: {joined}")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        rendered = " ".join(cmd)
        raise RuntimeError(
            f"Command failed: {rendered}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
