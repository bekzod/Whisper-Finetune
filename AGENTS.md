# Repository Guidelines

## Project Structure & Module Organization
- `finetune.py`: training entrypoint (Accelerate/DeepSpeed).  
- `evaluation.py`: batch evaluation and WER.  
- `infer.py`, `infer_server.py`, `infer_gui.py`: offline, API, and GUI inference.  
- `configs/`: training, datasets, and augmentation configs (e.g., `accelerate.yaml`, `datasets.json`).  
- `utils/`, `metrics/`, `tools/`: helpers, metric utilities, conversion scripts.  
- `run.sh`, `run-full.sh`, `run-eval.sh`: reproducible training/eval launchers.  
- `docs/`, `AndroidDemo/`, `WhisperDesktop/`, `static/`, `templates/`: documentation and UIs.  
- `tests/`: add unit/integration tests here.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`  
- Install deps: `pip install -r requirements.txt`  
- Configure accelerate: `accelerate config --config_file configs/accelerate.yaml`  
- Train (LoRA example): `bash run.sh` (uses `finetune.py` + `configs/datasets.json`).  
- Full finetune: `bash run-full.sh`  
- Evaluate: `bash run-eval.sh` or `python evaluation.py --help`  
- Inference (CLI): `python infer.py --help`; API: `python infer_server.py`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; prefer type hints for new/changed code.  
- Names: `snake_case` for files/functions/vars, `CamelCase` for classes, `UPPER_SNAKE` for constants.  
- Docstrings on public functions; keep modules focused and under ~500 lines.  
- Keep config in `configs/`; avoid hardcoding paths.

## Testing Guidelines
- Framework: `pytest` (add as a dev dependency if missing).  
- Location: `tests/` with `test_*.py`; mirror package/module names.  
- Cover data transforms, dataset loading, decoding, and metric computations.  
- Run: `pytest -q`; aim for coverage on core utils and critical paths.

## Commit & Pull Request Guidelines
- Current history is terse (e.g., "fixes", "Update X"). Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.  
- Commits: small, focused; include rationale in body if non-trivial.  
- PRs: clear description, what/why, configs changed, sample commands, before/after metrics (e.g., WER), and screenshots/logs. Link related issues.

## Security & Configuration Tips
- Do not commit secrets (e.g., `WANDB_API_KEY`, HF tokens). Use env vars: `export WANDB_API_KEY=...` and read via `os.environ`.  
- Keep dataset/model paths configurable; prefer `--args` or YAML.  
- Large artifacts: store outside repo; reference with paths in configs.

