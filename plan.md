# Plan / Working Memory

Last updated: 2026-02-12

This file is a lightweight experiment log + plan so future agents can pick up work without re-discovering context.

## Goal

- Make a convincing case that the FMM long-range block helps characterize long-range interactions on MD22 DWNT.
- Keep hybrid competitive on wall-clock speed (DDP) so any accuracy gains are not bought by an impractical slowdown.

## What Was Implemented

- Serial hybrid schedule support in the model:
  - `attn_type="first-order{K}+fmm-node{L}"` (e.g. `first-order6+fmm-node2`).
  - Runs K local (edge) layers first, then L global node-FMM layers, with standard residual connections between blocks.
- DDP Slurm script for serial hybrid:
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
  - Defaults to 4 GPUs and uses `torch.distributed.run` with `--nproc_per_node=$SLURM_JOB_GPUS` when available.
  - Local cutoff overrides supported via env vars:
    - `MAX_RADIUS`, `PBC_MAX_RADIUS`, `MAX_NEIGHBORS` (plumbed to Hydra `backbone_config.*`).
- FMM speed/accuracy knobs (Hydra):
  - `backbone_config.fmm_num_directions`
  - `backbone_config.fmm_kappa_chunk_size`
  - `backbone_config.fmm_compute_dtype` (`auto|fp32|bf16|fp16`)
- Value bottleneck to shrink the FMM value width:
  - New knob: `backbone_config.fmm_value_head_dim` (0 disables; e.g. 8 or 16).
  - Implemented in `src/molfm/models/e2former/fmm_e2former.py` as bias-free `v_proj/out_proj` around the FMM core.
  - Plumbed through `src/molfm/models/e2former/e2former.py` and `src/molfm/models/e2former/E2Former_configs.py`.
  - Defaults added to:
    - `config_file/backbone_config/e2former_fmm.yaml`
    - `config_file/backbone_config/e2former_hybrid.yaml`
- Optional pre-training speed bench integrated into the serial sbatch:
  - `RUN_FMM_VBOTTLENECK_BENCH=1`
  - `BENCH_VDIMS="0 8 16"` (etc.)
  - Runs `scripts/benchmark_e2former_fmm_variant.py` before training.
- Learnable spectral radial mixture for node-only/hybrid FMM:
  - `a_{lq}` coefficients in `AlphaFRYSphericalFMMMultiL` can now be trainable Parameters.
  - Added per-`l` coefficient support (`a_per_l=True`) so each spherical degree can use its own radial profile.
  - New Hydra knobs:
    - `backbone_config.fmm_learnable_radial_coeffs` (default `true`)
    - `backbone_config.fmm_radial_coeffs_mode` (`per_l_head|per_l_shared|head|shared`, default `per_l_head`)
    - `backbone_config.fmm_radial_init_scale` (default `0.05`)
    - `backbone_config.fmm_radial_low_kappa_bias` (default `2.0`)
  - Initialization is small and low-`kappa` biased to stabilize global-branch optimization and encourage short/long separation.

## Speed Benchmark (Value Bottleneck)

Source log: `outputs/slurm/dwnt_serial_v8_bench-1153689.out` (A40 node).

Benchmark settings:
- `layers=8`, serial `first-order6+fmm-node2`
- `B=2`, `nodes_per_graph=512`
- `nk=6`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`
- `radius=15.0`, `max_neighbors=20`, `tp_backend=cueq`

Results (forward time):

| fmm_value_head_dim | baseline(edge) | fmm-node | serial(6+2) |
|---:|---:|---:|---:|
| 0  | 79.581 ms | 148.662 ms | 99.529 ms |
| 8  | 79.549 ms | 67.612 ms  | 79.535 ms |
| 16 | 79.535 ms | 96.109 ms  | 86.522 ms |

Interpretation:
- The wide value width was a major bottleneck: `fmm-node` got ~2.20x faster going from vdim=0 -> 8.
- With `fmm_value_head_dim=8`, the serial hybrid is ~baseline speed in this microbench.

## Baseline Reference (MD22 DWNT)

Baseline short-range job completed:
- Job: `1153129` (log: `outputs/slurm/dwnt_e2former-1153129.out`)
- Final validation at `global_step=200000`:
  - `valid_loss=0.3093`
  - `force_loss=0.3836`

## Current Slurm Status (as of 2026-02-12)

Running:
- `1154481` `dwnt_serial_learnrad`: serial hybrid with learnable radial mixture enabled
  (`fmm_learnable_radial_coeffs=true`, `fmm_radial_coeffs_mode=per_l_head`,
  `fmm_radial_init_scale=0.05`, `fmm_radial_low_kappa_bias=2.0`,
  plus `fmm_value_head_dim=8`, `nk=6`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`).
- `1153471` `dwnt_serial_d16_bf16_nk4`: serial hybrid variant (nk=4).
- `1154468` `ood-vscode-proxy`: dev proxy.

Local (tmux):
- Session: `dwnt_local_learnrad_e3nn_20260211_223336`
- Log: `outputs/local_tmux/local_tmux_learnrad_e3nn_20260211_223336.log`
- Save dir: `outputs/runs/md22_dwnt/local_tmux_learnrad_e3nn_20260211_223336`
- Key overrides: `tp_type=QK_alpha+tp_e3nn`, `attn_type=first-order6+fmm-node2`,
  `fmm_learnable_radial_coeffs=true`, `fmm_radial_coeffs_mode=per_l_head`,
  `fmm_radial_init_scale=0.05`, `fmm_radial_low_kappa_bias=2.0`,
  `fmm_value_head_dim=8`, `nk=6`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`.

cuEquivariance import robustness (2026-02-12):
- Added timeout/retry guards to:
  - `scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
- New env knobs:
  - `CUEQ_IMPORT_TIMEOUT` (default `45`)
  - `CUEQ_IMPORT_RETRIES` (default `3`)
- Verified in tmux:
  - `cuequivariance_torch` imports successfully (`0.8.1`) and `_cueq_ops_available=True`.
  - Local cueq smoke run started successfully in session `dwnt_local_cueq_smoke_20260211_232236`
    with `tp_type=QK_alpha+tp_cueq` and learnable radial settings.

Held (queued but user-held):
- `1153474` `dwnt_serial_d09_bf16`
- `1153475` `dwnt_serial_d16_bf16_lr1e4`
- `1153531` `dwnt_serial_r7` (smaller local cutoff)
- `1153532` `dwnt_serial_r6` (smaller local cutoff)

Cancelled to free MaxGRESPerAccount quota:
- `1153472` `dwnt_serial_d25_fp32`
- `1153473` `dwnt_serial_d16_bf16`
- `1153689` `dwnt_serial_v8_bench` (manually cancelled on 2026-02-12 to free resources for learnable-radial run)

## Update (2026-02-12 00:20 EST)

Slurm failure root cause (priority-tier run):
- Job `1155005` (`dwnt_serial_learnrad_nmi`, `qos_nmi`) failed quickly.
- `sacct` state: `FAILED` with `ExitCode=1:0`.
- Error in `outputs/slurm/dwnt_serial_learnrad_nmi-1155005.err`:
  `torch.cuda.DeferredCudaCallError ... device=3, num_gpus=3`
  during `torch.cuda.set_device(args.local_rank)` for 4-rank launch.
- Interpretation: rank/GPU visibility mismatch on the allocated node for that run.

Local restart (tmux, single-GPU cueq + learnable radial):
- Session: `dwnt_local_cueq_learnrad_20260212_001857`
- Log: `outputs/local_tmux/dwnt_local_cueq_learnrad_20260212_001857.log`
- Save dir: `outputs/runs/md22_dwnt/dwnt_local_cueq_learnrad_20260212_001857`
- Launch mode: `torch.distributed.run --nproc_per_node=1` with
  `tp_type=QK_alpha+tp_cueq`, `attn_type=first-order6+fmm-node2`,
  `fmm_learnable_radial_coeffs=true`, `fmm_radial_coeffs_mode=per_l_head`,
  `fmm_radial_init_scale=0.05`, `fmm_radial_low_kappa_bias=2.0`.
- Runtime check: process is active (`train_molfm.py` PID `460307`);
  `nvidia-smi --query-compute-apps` reports GPU process and memory use (~15 GiB).

## Update (2026-02-12 00:40 EST)

Resume preempted learnrad run:
- Submitted resume job `1155009` (`dwnt_serial_learnrad_resume2`) from checkpoint:
  `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_serial_learnrad_20260211_204841/checkpoint_E92.pt`
- Current state: `PENDING (Resources)` with `QOS=normal`
- Added `--exclude=r4516u16n01` to avoid the prior node where NMI run hit the
  CUDA visibility mismatch.

## Update (2026-02-12 00:46 EST)

Cross-run analysis snapshot (from parsed logs):
- `dwnt_e2former-1153129` (short-range baseline): best `force_loss=0.3709`,
  last `0.3836`, `SamplePerSec~19.0` (best accuracy overall).
- `dwnt_serial_d16_bf16_nk4-1153471` (serial FMM, nk=4): best `force_loss=0.4649`,
  last `0.5254`, `SamplePerSec~7.65` (best serial/FMM quality so far).
- `dwnt_serial_v8_bench-1153689` (serial FMM, nk=6, v_head_dim=8): best
  `force_loss=0.5346`, worse quality than nk=4 run.
- `dwnt_serial_learnrad-1154481` (learnable radial, nk=6, v_head_dim=8):
  preempted early; best/last `force_loss=1.1529` at current training horizon.

New improved job launched (quality-oriented FMM):
- Job `1155139` (`dwnt_serial_nk4_learnrad`) submitted on `gpu` partition.
- Status: `PENDING (Priority)` at submission.
- Rationale: combine best serial setting (`nk=4`) with learnable radial mixture,
  while disabling value bottleneck (`fmm_value_head_dim=0`) to avoid the quality
  drop seen in the `v_head_dim=8` serial run.
- Key overrides:
  - `FMM_NUM_KAPPA=4`
  - `FMM_VALUE_HEAD_DIM=0`
  - `FMM_LEARNABLE_RADIAL_COEFFS=true`
  - `FMM_RADIAL_COEFFS_MODE=per_l_head`
  - `FMM_RADIAL_INIT_SCALE=0.05`
  - `FMM_RADIAL_LOW_KAPPA_BIAS=2.0`
  - `FMM_NUM_DIRECTIONS=16`
  - `FMM_COMPUTE_DTYPE=bf16`
  - `SEED=59`
  - `MAX_LR=5e-5`, `MIN_LR=5e-6`, `WARMUP_STEPS=2000`, `WEIGHT_DECAY=5e-3`
  - node exclude: `r4516u16n01`

Efficiency-focused comparison arm:
- Job `1155140` (`dwnt_serial_nk4_learnrad_v8`) submitted on `gpu` partition.
- Status: `PENDING (Priority)` at submission.
- Purpose: compare against `1155139` by increasing value bottleneck dimension
  for higher throughput expectation.
- Delta vs `1155139`:
  - `FMM_VALUE_HEAD_DIM=8` (instead of `0`)
- Kept fixed for fair comparison:
  - `FMM_NUM_KAPPA=4`, `FMM_NUM_DIRECTIONS=16`, `FMM_COMPUTE_DTYPE=bf16`
  - learnable radial settings (`per_l_head`, `init_scale=0.05`, low-kappa bias `2.0`)
  - optimizer/lr settings and `SEED=59`

## Update (2026-02-12 01:05 EST)

Queue/resource policy adjusted per request:
- Allowed mixes now target:
  - `4x` GPUs on `a100/h100`
  - `1x h200`
  - `2x h200`
- Cancelled non-matching `1x any-gpu` jobs:
  - `1155143`, `1155144`

Active runs after resubmission:
- `1155147` `dwnt_serial_nk4_learnrad_a100h100x4`
  - allocation: `gres/gpu:4` with excludes to avoid `a40/l40/h200` + problematic node
  - node: `r4519u13n01`
  - batch: `per_gpu=2`, global train batch `8`
  - key: `FMM_NUM_KAPPA=4`, `FMM_VALUE_HEAD_DIM=0`, learnable radial enabled
- `1155148` `dwnt_serial_nk4_learnrad_v8_h200x1`
  - allocation: `gres/gpu:h200:1`
  - node: `r818u33n06`
  - batch scaled: `per_gpu=8`, global train batch `8`
  - key: `FMM_NUM_KAPPA=4`, `FMM_VALUE_HEAD_DIM=8`, learnable radial enabled
- `1155145` `dwnt_serial_nk4_learnrad_v8_h200x2` (kept running)
  - allocation: `gres/gpu:h200:2`
  - node: `r818u29n04`
  - batch scaled: `per_gpu=4`, global train batch `8`
  - key: `FMM_NUM_KAPPA=4`, `FMM_VALUE_HEAD_DIM=8`, learnable radial enabled

Local run (requested batch size 8 + improved hyperparameter):
- Session: `dwnt_local_bs8_nk4_20260212_011625`
- Log: `outputs/local_tmux/dwnt_local_bs8_nk4_20260212_011625.log`
- Save dir: `outputs/runs/md22_dwnt/dwnt_local_bs8_nk4_20260212_011625`
- Launch: `torch.distributed.run --nproc_per_node=1` (single local H100)
- Batch: `train_batch_size=8`, `val_batch_size=8` (global batch 8)
- Suggested improved hyperparameter applied: `fmm_num_kappa=4`
  (kept learnable radial mixture enabled and `fmm_value_head_dim=0`).

Resume launch request (2026-02-12):
- Requested resume target: `dwnt_serial_learnrad_20260211_204841`
- Submitted resume job: `1155619` (`dwnt_serial_learnrad_resume`)
- State after submit: `RUNNING`
- Checkpoint used:
  `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_serial_learnrad_20260211_204841/checkpoint_E92.pt`
- Save dir reused:
  `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_serial_learnrad_20260211_204841`
- Key resume overrides to match checkpoint architecture:
  - `FMM_NUM_KAPPA=6`
  - `FMM_VALUE_HEAD_DIM=8`
  - `FMM_LEARNABLE_RADIAL_COEFFS=true`
  - `FMM_RADIAL_COEFFS_MODE=per_l_head`
  - `FMM_NUM_DIRECTIONS=16`
  - `FMM_COMPUTE_DTYPE=bf16`

Hyperparameter tuning sweep (2026-02-12, a100/h100 4-GPU profile):
- Submitted:
  - `1155641` `dwnt_tune_d12` (`WANDB_RUN_NAME=dwnt_tune_d12_20260212_093649`)
    - `FMM_NUM_DIRECTIONS=12` (speed-leaning variant)
  - `1155642` `dwnt_tune_d24` (`WANDB_RUN_NAME=dwnt_tune_d24_20260212_093649`)
    - `FMM_NUM_DIRECTIONS=24` (quality-leaning variant)
  - `1155643` `dwnt_tune_lr3e5_radial` (`WANDB_RUN_NAME=dwnt_tune_lr3e5_radial_20260212_093649`)
    - `FMM_NUM_DIRECTIONS=16`
    - `MAX_LR=3e-5`, `MIN_LR=3e-6`
    - `FMM_RADIAL_INIT_SCALE=0.02`
    - `FMM_RADIAL_LOW_KAPPA_BIAS=3.0`
- Shared sweep baseline settings:
  - `FMM_NUM_KAPPA=4`, `FMM_KAPPA_MIN=0.8`, `FMM_KAPPA_MAX=1.2`
  - `FMM_VALUE_HEAD_DIM=0`
  - learnable radial enabled (`per_l_head`)
  - `PER_GPU_BATCH=2`, `PER_GPU_VAL_BATCH=2` (global batch 8 on 4 GPUs)
  - `FMM_COMPUTE_DTYPE=bf16`, `SEED=59`
- Current status after submit:
  - `1155641` RUNNING (`r4519u04n01`)
  - `1155642` RUNNING (`r4519u10n01`)
  - `1155643` PENDING (`Resources`)

## Update (2026-02-12 21:00 EST)

Freed low-value runs (per latest validation trends) to release GPUs:
- Cancelled:
  - `1155641` `dwnt_tune_d12`
  - `1155642` `dwnt_tune_d24`
  - `1155643` `dwnt_tune_lr3e5_radial`
  - `1155619` `dwnt_serial_learnrad_resume`
  - `1155147` `dwnt_serial_nk4_learnrad_a100h100x4`
- Kept running (best current FMM trajectories):
  - `1155145` `dwnt_serial_nk4_learnrad_v8_h200x2`
  - `1155148` `dwnt_serial_nk4_learnrad_v8_h200x1`

Submitted new from-scratch full-length runs (`TOTAL_NUM_STEPS=200000`, `2x h200`, global batch `8`):
- `1156470` `dwnt_fs_ctrl`
  - `FMM_NUM_KAPPA=4`, `FMM_VALUE_HEAD_DIM=8`, `FMM_NUM_DIRECTIONS=16`
  - `MAX_LR=5e-5`, `MIN_LR=5e-6`, `WARMUP_STEPS=2000`, `SEED=59`
  - `WANDB_RUN_NAME=dwnt_fs_ctrl_20260212_210055`
- `1156471` `dwnt_fs_lr4e5`
  - same as control except `MAX_LR=4e-5`, `MIN_LR=4e-6`
  - `WANDB_RUN_NAME=dwnt_fs_lr4e5_20260212_210055`
- `1156472` `dwnt_fs_v16`
  - same as control except `FMM_VALUE_HEAD_DIM=16`
  - `WANDB_RUN_NAME=dwnt_fs_v16_20260212_210055`
- Submit-time scheduler state:
  - all three jobs entered `PENDING` with `Reason=None`
  - `ReqTRES=cpu=12,mem=96G,node=1,gres/gpu:h200=2`

## Next Steps

1. Accuracy check for the speed win:
   - Compare validation force MAE of serial hybrid with `FMM_VALUE_HEAD_DIM=8` vs:
     - baseline (short-range)
     - serial hybrid without bottleneck (`FMM_VALUE_HEAD_DIM=0`)
2. Cutoff sweep for speed/accuracy:
   - Run/release the pending `r6/r7` serial jobs and measure both:
     - Samples/sec
     - Force MAE
3. If still too slow or unstable:
   - Reduce `fmm_num_directions` (fast) and/or tighten kappa band (accuracy/equivariance).
   - Consider a scalar-only FMM path (l=0 only) if we need an even faster long-range branch.

## Repro Commands

Local GPU microbench (single node, no Slurm):

```bash
PYTHONPATH=./src python scripts/benchmark_e2former_fmm_variant.py \
  --device cuda \
  --B 2 --nodes-per-graph 512 --layers 8 \
  --include-serial --serial-local-layers 6 \
  --radius 15.0 --max-neighbors 20 --pos-scale 1.0 \
  --fmm-tp-backend cueq --fmm-num-kappa 6 --fmm-kappa-min 0.8 --fmm-kappa-max 1.2 \
  --fmm-num-directions 16 --fmm-compute-dtype bf16 \
  --fmm-value-head-dim 8
```

Slurm serial hybrid (value bottleneck):

```bash
sbatch --job-name=dwnt_serial_v8 \
  --export=ALL,FMM_VALUE_HEAD_DIM=8 \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch
```

Slurm serial hybrid (learnable radial mixture, explicit overrides):

```bash
sbatch --job-name=dwnt_serial_learnrad \
  --export=ALL,FMM_LEARNABLE_RADIAL_COEFFS=true,FMM_RADIAL_COEFFS_MODE=per_l_head,FMM_RADIAL_INIT_SCALE=0.05,FMM_RADIAL_LOW_KAPPA_BIAS=2.0,FMM_VALUE_HEAD_DIM=8,FMM_NUM_KAPPA=6,FMM_KAPPA_MIN=0.8,FMM_KAPPA_MAX=1.2,FMM_NUM_DIRECTIONS=16,FMM_COMPUTE_DTYPE=bf16 \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch
```
