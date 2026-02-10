# Repository Guidelines

## Project Structure & Module Organization

- `src/molfm/`: main Python package
  - `models/` (E2Former + heads), `data/` (LMDB datasets/collators), `pipeline/` (training loop), `tasks/` (entrypoints), `utils/`, `logging/`
- `config_file/`: Hydra configs (e.g., `config_molfm.yaml`, `backbone_config/e2former.yaml`)
- `scripts/`: launchers (currently `train_e2former.sh`)
- `resources/`: non-code assets (e.g., `main.png`)
- Generated artifacts are ignored by git: `outputs/`, `wandb/`, `swanlog/`, `__pycache__/` (see `.gitignore`).

## Build, Test, and Development Commands

This repo is run directly from source (no `setup.py`/`pyproject.toml`).

```bash
conda env create -f environment.yaml
conda activate e2former
pip install fairchem-core==1.3.0 --no-deps
export PYTHONPATH=./src:$PYTHONPATH
```

Run training via the provided launcher (expects dataset paths via env vars):

```bash
data_path=/path/to/lmdb_root save_dir=./outputs/run1 \
(cd src && bash ../scripts/train_e2former.sh)
```

Direct entrypoint (Hydra loads `config_file/config_molfm.yaml` by default):
`torchrun --nproc_per_node=1 src/molfm/tasks/train_molfm.py`.

## Coding Style & Naming Conventions

- Python: 4-space indentation; `snake_case` for functions/vars, `PascalCase` for classes.
- Keep changes minimal and consistent with existing patterns (logging via `molfm.logging`, configs via Hydra).
- Imports: a few files use `# isort:skip`; avoid removing these unless you’re also fixing import ordering.

## Testing Guidelines

There is no dedicated unit-test suite. Preferred sanity checks:

- `PYTHONPATH=./src python -m compileall src`
- Run a short `torchrun` job on a tiny LMDB shard to validate end-to-end wiring.

## Commit & Pull Request Guidelines

- Existing history uses short, imperative subjects (e.g., `Update config`, `Update README.md`).
- PRs should include: a brief motivation, the exact command/config overrides used (Hydra), and any expected training/logging impact (W&B/SwanLab).

## Agent-Specific Notes (Optional)

- Avoid formatting-only churn and large refactors unless requested.
- If you change launch scripts or configs, update `README.md` and keep `.gitignore` aligned with new artifacts.
