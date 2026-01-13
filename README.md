# UBio-MolFM/E2Former-LSR

<p align="center">
  <img src="./resources/main.png" alt="Architectural necessity and benchmark scope for long-range MLFFs" width="800"/>
</p>

This repo contains the code for the [Scalable Machine Learning Force Fields for Macromolecular Systems Through Long-Range Aware Message Passing](https://arxiv.org/abs/2601.03774).

At its core, E2Former utilizes **Wigner 6j convolution** for efficient and accurate tensor product operations, enabling the model to capture complex geometric interactions while preserving physical symmetries. Besides, we developed E2former-LSR, a unified SO(3)-equivariant neural architecture that integrates Long–Short Range Message Passing (LSR-MP) with an E2Former backbone to capture both local and nonlocal interactions in molecular systems.

## Overview

This repo contains the E2Former-LSR training pipeline, datasets utilities, and a runnable
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

## Common Entry Points

- `src/molfm/tasks/train_molfm.py`: main training entrypoint
- `scripts/train_e2former_test.sh`: test training launcher

## Data and Checkpoints

We provide the dataset and pretrained checkpoints used in our paper at:
`https://huggingface.co/datasets/IQuestLab/UBio-MolLR25`. The release includes data
files and ready-to-use checkpoints for inference and fine-tuning.

## Troubleshooting

- Ensure your LMDB dataset paths exist and are accessible by the script.
- If using distributed training, confirm `MASTER_ADDR`, `MASTER_PORT`, and
  related environment variables are set correctly.
