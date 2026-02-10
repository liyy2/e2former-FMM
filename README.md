# E2Former-FMM


## Overview

This repo contains the E2Former-FMM training pipeline, datasets utilities, and a runnable
training entrypoint. The default training shell script is `scripts/train_e2former.sh`,
which launches the Python entrypoint in `src/molfm/tasks/train_molfm.py`.

## Project Layout

- `src/molfm/models`: backbone and head definitions for E2Former and related models
- `src/molfm/data`: dataset loaders, batching, and collators
- `src/molfm/pipeline`: training loop, engine abstraction, and training schema
- `src/molfm/utils`: runtime helpers and optimizer tools
- `scripts`: runnable shell scripts for training
- `config_file`: configuration files for training runs

## Quick Start

Create the environment and install dependencies:

```bash
conda env create -f environment.yaml
conda activate e2former
pip install fairchem-core==1.3.0 --no-deps
export PYTHONPATH=./src:$PYTHONPATH
```

Run the default training script:
Noted: Set inference_mode=True for checkpoint inference, inference_mode=False for training.

```bash
bash scripts/train_e2former.sh
```

`data_path_list` is the training dataset path(s), and `data_path_list_valid` is the
validation dataset path(s). Both are configured inside `scripts/train_e2former.sh`
and can be overridden via environment variables.

## Configuration Notes

`scripts/train_e2former.sh` sets all required runtime options, including dataset paths,
optimizer schedule, and E2Former backbone parameters. You can override values by exporting
environment variables before invoking the script.

Node-only FMM variant:

- Use `config_file/backbone_config/e2former_fmm.yaml` as the Hydra backbone config.
- This variant enables `attn_type: fmm-node` and `tp_type: fmm-node+tp_cueq`, which runs
  node-based FMM attention without edge-neighbor attention construction and uses
  cuEquivariance tensor-product kernels when available.
- The default FMM radial band is configured by `fmm_num_kappa`, `fmm_kappa_min`,
  and `fmm_kappa_max` in the same config; the shipped defaults prioritize lower
  equivariance error at unchanged compute complexity.

Hybrid short+long variant:

- Use `config_file/backbone_config/e2former_hybrid.yaml`.
- This runs local edge-based E2Former attention and global node-based FMM attention
  in parallel and fuses them as `local + long_scale * global`.
- Set `attn_type: hybrid-<local_type>` (for example `hybrid-first-order`).
- Set `tp_type` as `<local_tp>@<fmm_tp>`; for example
  `QK_alpha@fmm-node+tp_cueq`.

MD22 baseline vs baseline+FFM protocol:

- Use `scripts/train_md22_baseline_vs_hybrid.sh` to run both variants with identical
  splits and optimization settings.
- The split follows the LSR-MP style protocol: for each molecule, choose a
  molecule-specific `sample_size`; split that subset into `train/val` by
  `md22_train_prop` (default `0.95`), and use the remainder of the full dataset
  as test.
- Default molecule `sample_size` values are:
  `AT_AT=3000`, `AT_AT_CG_CG=2000`, `stachyose=8000`, `DHA=8000`,
  `Ac_Ala3_NHMe=6000`, `buckyball_catcher=600`,
  `double_walled_nanotube=800`, `chig=8000`.
- Override with `md22_sample_size=<int>` if needed.

Example:

```bash
data_path=/path/to/md22 \
data_path_list=chig/radius3/chig.lmdb \
dataset_name_list=md \
md22_molecule=chig \
save_root=./outputs/md22_chig \
bash scripts/train_md22_baseline_vs_hybrid.sh
```

## Common Entry Points

- `src/molfm/tasks/train_molfm.py`: main training entrypoint
- `scripts/train_e2former_test.sh`: test training launcher
- `scripts/benchmark_e2former_fmm_variant.py`: baseline edge-attention vs node-only FMM speed benchmark

## Data and Checkpoints

We provide the dataset and pretrained checkpoints used in our paper at:
`https://huggingface.co/datasets/IQuestLab/UBio-MolLR25`. The release includes data
files and ready-to-use checkpoints for inference and fine-tuning.

## Troubleshooting

- Ensure your LMDB dataset paths exist and are accessible by the script.
- If using distributed training, confirm `MASTER_ADDR`, `MASTER_PORT`, and
  related environment variables are set correctly.
