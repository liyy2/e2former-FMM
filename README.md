# E2Former-FMM


## Overview

This repo contains the E2Former-FMM training pipeline, datasets utilities, and a runnable
training entrypoint. The default training shell script is `scripts/train_e2former.sh`,
which launches the Python entrypoint in `src/molfm/tasks/train_molfm.py`.

The FMM variants are designed to capture **global context** efficiently: instead of building
explicit pairwise edges for all node pairs, they aggregate global moments and evaluate each
node with linear-time node-wise operations.

## FMM Method Summary (aligned with `FMM/fmm.tex`)

The core operator follows an equivariant attention form:

$$
\mathbf m_i^{(L)}=\sum_j \alpha_{ij}\,[\mathbf v_j^{(\lambda)}\otimes (f_\ell(r_{ij})\mathbf Y^{(\ell)}(\hat{\mathbf r}_{ij}))]^{(L)}.
$$

The implementation uses three key approximations/factorizations described in the TeX writeup:

1. Linearized attention weights:
$$
\alpha_{ij}\approx \frac{\langle \varphi(\mathbf q_i),\varphi(\mathbf k_j)\rangle}{Z_i},\quad
Z_i=\sum_n\langle \varphi(\mathbf q_i),\varphi(\mathbf k_n)\rangle.
$$

2. Spectral radial expansion for each angular order $\ell$:
$$
f_\ell(r)\approx \sum_{q=1}^{Q} a_{\ell q}\,j_\ell(\kappa_q r).
$$
The mixture coefficients $a_{\ell q}$ are learnable in the node-FMM branch (default:
per-$\ell$ per-head), initialized with small magnitude and a low-$\kappa$ bias.

3. Plane-wave/spherical quadrature factorization:
$$
j_\ell(\kappa r)Y_\ell(\hat r)\ \leadsto\ \sum_s w_s\,Y_\ell(u_s)\,e^{i\kappa u_s\cdot r_i}\,e^{-i\kappa u_s\cdot r_j}.
$$

This yields a true node-wise form where all $j$-dependence is compressed into global moments
$\mathbf M_{q,s}$ and a global key sum, then each node $i$ is evaluated without explicit edge loops.

Why this is useful:

- Captures long-range/global interactions in a single layer (not limited by local neighbor cutoff).
- Keeps equivariant angular structure through spherical harmonics and tensor coupling.
- Reduces global interaction cost from quadratic pairwise aggregation to linear-time moment aggregation/evaluation (for fixed spectral and quadrature ranks).

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
  node-based FMM attention without edge-neighbor attention construction, captures global
  context through moment aggregation, and uses
  cuEquivariance tensor-product kernels when available.
- The default FMM radial band is configured by `fmm_num_kappa`, `fmm_kappa_min`,
  and `fmm_kappa_max` in the same config; the shipped defaults prioritize lower
  equivariance error at unchanged compute complexity.
- Radial-mixture learning knobs:
  `fmm_learnable_radial_coeffs`, `fmm_radial_coeffs_mode`,
  `fmm_radial_init_scale`, `fmm_radial_low_kappa_bias`.

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
