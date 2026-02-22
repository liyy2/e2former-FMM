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

## Update (2026-02-15)

Storage quota cleanup after checkpoint write failures:
- Root cause of recent job failures was confirmed as storage quota pressure while writing checkpoints (`PytorchStreamWriter failed writing file ...`, `Disk quota exceeded` on larger writes in `outputs/runs`).
- Cleanup actions applied under `outputs/runs/md22_dwnt`:
  - For run dirs with multiple `checkpoint_E*.pt`, kept only:
    - latest epoch checkpoint
    - `checkpoint_best.pt` target when present
  - Rewrote `checkpoint_list.txt` in each pruned run dir to match remaining checkpoints.
  - Removed older dated run directories (`YYYYMMDD < 20260212`) while preserving baseline-related paths.
- Cleanup summary:
  - `pruned_dirs=42`
  - `removed_ckpt_files=16463`
  - `removed_run_dirs=26`
  - size reduced from `~1005G` to `~1.7G` in `outputs/runs/md22_dwnt`
  - reclaimed bytes: `1,076,448,901,180`
- Post-clean validation:
  - large write test in `outputs/runs` succeeds (256 MB).

## Update (2026-02-15) - Periodic test-set evaluation

Implemented step-based test-set evaluation (distinct from validation):
- Added new run config knob:
  - `test_batch_interval` (default `20000` in `config_file/config_molfm.yaml`)
- Pipeline changes:
  - `TrainingLoop` now supports periodic `test()` calls during training when
    `global_step % test_batch_interval == 0`.
  - Added `test_data_loader` plumbing through execution engines.
  - Added shared `_evaluate(..., split)` path used by both `validate()` and `test()`.
- Slurm launcher defaults updated for MD22 FMM/hybrid workflows:
  - `scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
  - New env overrides:
    - `TEST_BATCH_INTERVAL` (default `20000`)
    - `VAL_BATCH_INTERVAL` (default `0`)
    - `VAL_EPOCH_INTERVAL` (default `0`)
  - These defaults disable periodic validation and use test-set checks at large step intervals.

## Update (2026-02-15) - New FMM-strengthening Slurm matrix

Reasoning from prior findings:
- Serial 6+2 audit showed FMM branch is active but much smaller than local branch amplitude.
- Likely improvements: reduce local dominance, increase global-depth usage, improve FMM numerical fidelity, and strengthen low-kappa radial initialization.

Operational safeguard:
- Added and used `save_epoch_interval=50` plus `test_batch_interval=20000` and disabled val-interval eval (`val_batch_interval=0`, `val_epoch_interval=0`) to avoid checkpoint-quota blowups.

Replaced earlier quick-start jobs:
- Cancelled: `1160097`, `1160098`, `1160099` (launched before the save-interval safeguard).

Submitted jobs (all running at submit check):
- `1160100` `dwnt_s6g2_count_ctrl`
  - W&B: `dwnt_s6g2_count_ctrl_20260215_121259`
  - Control: `first-order6+fmm-node2`, `nk=4`, `v_head_dim=8`, `coupling_norm=count`, `radius=15`, `neighbors=20`.
- `1160101` `dwnt_s5g3_count`
  - W&B: `dwnt_s5g3_count_20260215_121259`
  - More global depth: `first-order5+fmm-node3` (same remaining knobs as control).
- `1160102` `dwnt_s6g2_count_r10`
  - W&B: `dwnt_s6g2_count_r10_20260215_121259`
  - Weaker local branch: `radius=10`, `max_neighbors=24` (other knobs control-like).
- `1160103` `dwnt_s6g2_count_fp32`
  - W&B: `dwnt_s6g2_count_fp32_20260215_121259`
  - Higher FMM fidelity: `fmm_compute_dtype=fp32`, stable LR (`4e-5/4e-6`), warmup `4000`.
- `1160104` `dwnt_s6g2_count_radboost`
  - W&B: `dwnt_s6g2_count_radboost_20260215_121259`
  - Stronger low-kappa radial signal: `fmm_radial_init_scale=0.1`, `fmm_radial_low_kappa_bias=3.0`, stable LR (`4e-5/4e-6`), warmup `4000`.

Shared launch context for this matrix:
- Script: `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
- `TOTAL_NUM_STEPS=100000`, `SAVE_EPOCH_INTERVAL=50`, `TEST_BATCH_INTERVAL=20000`.
- Excluded low-end GPU nodes (`a40/l40s` family) to reduce hardware variance.

## Update (2026-02-15) - Validation default made frequent again

Adjusted Slurm launcher defaults to restore frequent epoch validation while keeping step-based test:
- Updated:
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch`
- New defaults:
  - `VAL_EPOCH_INTERVAL=10` (was `0`)
  - `VAL_BATCH_INTERVAL=0` (unchanged)
  - `TEST_BATCH_INTERVAL=20000` (unchanged)
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

## Update (2026-02-13) - Serial FMM Branch Magnitude Audit

Run inspected:
- `dwnt_serial_nk4_learnrad_v8_h200x2_20260212_010116`
  - local W&B run dir: `wandb/run-20260212_010214-uqo196t3`
  - best by `force_loss` in log: `checkpoint_E956.pt` (`force_loss=0.428529`)
  - best by `valid_loss` in log: `checkpoint_E988.pt` (`valid_loss=0.351276`)
  - latest inspected: `checkpoint_E1067.pt`

Findings from per-layer norm probe on a validation sample (`num_atoms=370`):
- This run is serial (`attn_type=first-order6+fmm-node2`), so there is no hybrid `long_scale` gate parameter in checkpoint.
- Local layers (`0..5`) attention branch magnitude (`ga_out`) RMS: ~`0.042` average.
- FMM layers (`6,7`) attention branch magnitude RMS: ~`0.0030` to `0.0041`.
  - Relative to local branch reference: `~0.07` to `0.10` (about 10-14x smaller).
  - Relative to residual stream RMS at layers 6-7: `ga/residual ~0.006-0.007`.
- Norm placement did not show collapse of input scale before FMM:
  - `norm1_out/residual` at layers 6-7 stayed around `1.6-1.8` (not near zero).
- FMM radial coefficient tensor magnitude was not near-zero:
  - `blocks.6.ga.fmm_multi_l.a` mean `|a| ~ 0.13`
  - `blocks.7.ga.fmm_multi_l.a` mean `|a| ~ 0.11`

Interpretation:
- The FMM branch is active but very weak in amplitude versus local attention in this serial setup.
- The issue is unlikely to be a stuck zero-initialized scalar gate in this run (none exists in serial mode).

## Update (2026-02-13) - Added FMM coupling normalization mode

- Added new knob: `backbone_config.fmm_coupling_norm` with options:
  - `count` (existing behavior, divide by coupling count)
  - `sqrt` (divide by sqrt of coupling count)
  - `none` (no coupling-count normalization)
- Wired through:
  - `src/molfm/models/e2former/fmm_e2former.py`
  - `src/molfm/models/e2former/e2former.py`
  - `config_file/backbone_config/e2former_fmm.yaml`
  - `config_file/backbone_config/e2former_hybrid.yaml`
  - Slurm launchers + benchmark helper for easy override.

Reproduction commands:

```bash
# Serial hybrid run with sqrt coupling normalization
sbatch --job-name=dwnt_serial_sqrt \
  --export=ALL,FMM_COUPLING_NORM=sqrt \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch

# FMM-only run with sqrt coupling normalization
sbatch --job-name=dwnt_fmm_sqrt \
  --export=ALL,FMM_COUPLING_NORM=sqrt \
  scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch
```

## Update (2026-02-12 22:40 EST) - No-Norm run launched

- Submitted no-norm serial run:
  - Job ID: `1156744`
  - Job name: `dwnt_serial_nonorm`
  - Scheduler state at submit check: `PD (Priority)`
  - Override: `FMM_COUPLING_NORM=none`
  - W&B run name: `dwnt_serial_nonorm_20260212_224026`
- Submit command:

```bash
sbatch --job-name=dwnt_serial_nonorm \
  --export=ALL,FMM_COUPLING_NORM=none,WANDB_RUN_NAME=dwnt_serial_nonorm_20260212_224026 \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch
```

## Update (2026-02-12 22:40 EST) - Sqrt run launched + default switched

- Submitted sqrt serial run:
  - Job ID: `1156749`
  - Job name: `dwnt_serial_sqrt`
  - Initial state: `R` on `r4518u09n01`
  - Override: `FMM_COUPLING_NORM=sqrt`
  - W&B run name: `dwnt_serial_sqrt_20260212_224215`
- Submit command:

```bash
sbatch --job-name=dwnt_serial_sqrt \
  --export=ALL,FMM_COUPLING_NORM=sqrt,WANDB_RUN_NAME=dwnt_serial_sqrt_20260212_224215 \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch
```

- Changed defaults to `sqrt` for future runs:
  - `src/molfm/models/e2former/fmm_e2former.py`
  - `src/molfm/models/e2former/e2former.py`
  - `config_file/backbone_config/e2former_fmm.yaml`
  - `config_file/backbone_config/e2former_hybrid.yaml`
  - `scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_cueq.sbatch`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
  - `scripts/benchmark_e2former_fmm_variant.py`

## Update (2026-02-13) - Removed forced fp32 Q/K cast in FMM

- In `src/molfm/models/e2former/fmm_e2former.py`, removed the explicit
  `packed_irreps.to(torch.float32)` before `q_proj/k_proj`.
- Q/K now use the incoming dtype so AMP/autocast can control precision.

## Update (2026-02-13) - Default `fmm_value_head_dim` set to 8

- Switched the default value bottleneck from `0` to `8` for future FMM/hybrid runs.
- Updated in:
  - `src/molfm/models/e2former/fmm_e2former.py`
  - `src/molfm/models/e2former/e2former.py`
  - `src/molfm/models/e2former/E2Former_configs.py`
  - `config_file/backbone_config/e2former_fmm.yaml`
  - `config_file/backbone_config/e2former_hybrid.yaml`
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
  - `scripts/benchmark_e2former_fmm_variant.py`

## Update (2026-02-15) - New 6+2 serial-hybrid stability/quality matrix launched

Evidence-based hypothesis (from current `md22_dwnt` runs):
- Best finished 6+2 run remains `2otro7n7` with `nk=4`, `v_head_dim=8`, and legacy/default coupling behavior (pre-`sqrt` default switch).
- Post-switch `sqrt`/`none` runs (`mgoyn7qw`, `39xsu6kv`) underperformed and failed earlier, but `sqrt + nk=4` families showed some upside when stabilized.
- Main risk appears to be late-stage instability rather than immediate divergence.

Code-side launcher improvement:
- Added `HYBRID_LONG_SCALE_INIT` override support to:
  - `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch`
- Plumbed to Hydra:
  - `backbone_config.hybrid_long_scale_init="${HYBRID_LONG_SCALE_INIT}"`

Submitted Slurm runs (all currently `RUNNING` at submit check):
- `1160080` `dwnt_s6g2_count_ctrl`
  - W&B: `dwnt_s6g2_count_ctrl_20260215_105352`
  - Purpose: control arm close to best known 6+2 recipe.
  - Overrides: `nk=4`, `v_head_dim=8`, `coupling_norm=count`, `lr=5e-5/5e-6`, `radius=15`, `neighbors=20`, `hybrid_long_scale_init=1.0`.
- `1160081` `dwnt_s6g2_sqrt_stable`
  - W&B: `dwnt_s6g2_sqrt_stable_20260215_105352`
  - Purpose: test whether `sqrt` can work with extra stability bias.
  - Overrides: `nk=4`, `v_head_dim=8`, `coupling_norm=sqrt`, `lr=4e-5/4e-6`, `warmup=4000`, `hybrid_long_scale_init=0.5`.
- `1160082` `dwnt_s6g2_count_r12`
  - W&B: `dwnt_s6g2_count_r12_20260215_105352`
  - Purpose: force more long-range burden onto FMM.
  - Overrides: `nk=4`, `v_head_dim=8`, `coupling_norm=count`, `radius=12`, `pbc_radius=12`, `max_neighbors=32`, `lr=5e-5/5e-6`.
- `1160083` `dwnt_s6g2_count_v16`
  - W&B: `dwnt_s6g2_count_v16_20260215_105352`
  - Purpose: test higher FMM value capacity with stability-leaning LR.
  - Overrides: `nk=4`, `v_head_dim=16`, `coupling_norm=count`, `lr=4e-5/4e-6`, `warmup=4000`, `hybrid_long_scale_init=0.7`.

Local tmux run:
- First launch attempt failed immediately due DeepSpeed CUDA op probe:
  - Error: `MissingCUDAException: CUDA_HOME does not exist`.
- Relaunched successfully with `DS_ACCELERATOR=cpu` (matching Slurm scripts):
  - Session: `dwnt_local_s6g2_count_r12_20260215_105527`
  - W&B: `dwnt_local_s6g2_count_r12_20260215_105527`
  - Log: `outputs/local_tmux/dwnt_local_s6g2_count_r12_20260215_105527.log`
  - Save dir: `outputs/runs/md22_dwnt/local_tmux/dwnt_local_s6g2_count_r12_20260215_105527`
  - Config: `first-order6+fmm-node2`, `nk=4`, `v_head_dim=8`, `coupling_norm=coreunt`,
    `radius=12`, `max_neighbors=32`, `train_batch_size=8`, `total_num_steps=80000`, `seed=59`.

## Update (2026-02-18) - True Resume Relaunch + From-Scratch LR Sweep

Queue triage snapshot:
- `gpu` partition currently had idle `a100` node(s), so submissions were pinned to `--gres=gpu:a100:4` for faster start.

Stateful resume intent:
- Initial submissions `1166122/1166123/1166124` used `LOADCHECK_PATH` and started from loaded weights, but did **not** restore optimizer/global-step state (`Start Training for epoch: 0`).
- These were canceled and replaced with true stateful resumes by reusing the original `SAVE_DIR` that already contains `checkpoint_list.txt`.

True stateful resume jobs:
- `1166128` `dwnt_r10_true_resume`
  - Save dir reused: `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_s6g2_count_r10_20260215_121259`
  - Target: extend to `TOTAL_NUM_STEPS=160000`
  - Key knobs: `radius=10`, `neighbors=24`, `nk=4`, `v_head_dim=8`, `coupling_norm=count`, `lr=5e-5/5e-6`
- `1166129` `dwnt_r15_true_resume`
  - Save dir reused: `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_s6g2_count_ctrl_20260215_121259`
  - Target: extend to `TOTAL_NUM_STEPS=160000`
  - Key knobs: `radius=15`, `neighbors=20`, `nk=4`, `v_head_dim=8`, `coupling_norm=count`, `lr=5e-5/5e-6`

From-scratch LR sweep (r10 recipe):
- `1166125` `dwnt_r10_fs_lr45` (`MAX_LR=4.5e-5`, `MIN_LR=4.5e-6`)
- `1166126` `dwnt_r10_fs_lr50` (`MAX_LR=5.0e-5`, `MIN_LR=5.0e-6`)
- `1166127` `dwnt_r10_fs_lr55` (`MAX_LR=5.5e-5`, `MIN_LR=5.5e-6`)
- Shared sweep setup:
  - `TOTAL_NUM_STEPS=80000`, `SAVE_EPOCH_INTERVAL=100`, `VAL_EPOCH_INTERVAL=0`, `TEST_BATCH_INTERVAL=20000`
  - `first-order6+fmm-node2`, `radius=10`, `neighbors=24`
  - `nk=4`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`, `v_head_dim=8`, `coupling_norm=count`, `seed=59`

Local tmux true resume:
- Session: `dwnt_local_r10_state_resume_20260218_013331`
- Log: `outputs/local_tmux/dwnt_local_r10_state_resume_20260218_013331.log`
- Save dir: `outputs/runs/md22_dwnt/local_tmux/dwnt_local_r10_state_resume_20260218_013331`
- Bootstrap for stateful local resume:
  - copied `checkpoint_E900.pt` from the r10 run into local save dir
  - wrote `checkpoint_list.txt` with `checkpoint_E900.pt`
  - launched with `ifresume=True`, observed:
    - `Resume from checkpoint: .../checkpoint_E900.pt`
    - `Start Training for epoch: 900`

## Update (2026-02-18 13:02 EST) - Recent W&B analysis + best-run stateful resume

Recent-run scope and normalization:
- W&B scope: entity/project/group = `yl2428/ffm/md22_dwnt`, `createdAt > 2026-02-18T00:00:00Z`.
- Primary objective: minimize `test/force_loss` when available (evaluation-aligned).
- Fallback comparison for active from-scratch LR sweep used normalized horizon at `_step<=6555`
  (same effective batch across runs): `min(train/loss)` up to horizon.

Recent run snapshot:
- `1a6ufx5k` (`dwnt_local_r10_state_resume_20260218_013331`, crashed):
  `test/force_loss=0.4589`, `test/valid_loss=0.3695`, `_step=145635` (best available eval).
- `ttkxc4rs` (`dwnt_r10_fs_lr45_20260218_013050`, running):
  `test/force_loss=0.9065`, `_step=27075`.
- `1ca5w93l` (`dwnt_r10_fs_lr50_20260218_013050`, running):
  `_step=6650`, no test metric yet; normalized `_step<=6555` `min(train/loss)~1.3777`.
- `uc9610fp` (`dwnt_r10_fs_lr55_20260218_013050`, running):
  `_step=285` (too early/incomplete for ranking).

Chosen resume target (best recent eval run):
- Run id: `1a6ufx5k`
- Save dir (reused for true stateful resume):
  `outputs/runs/md22_dwnt/local_tmux/dwnt_local_r10_state_resume_20260218_013331`
- Latest checkpoint in `checkpoint_list.txt`: `checkpoint_E1500.pt`

Submitted resume job:
- Job ID: `1166968`
- Job name: `dwnt_best_resume`
- W&B run name: `dwnt_r10_best_resume_20260218_130152`
- State after submit check: `PENDING (QOSMaxGRESPerUser)`
- Submit command:
```bash
sbatch --job-name=dwnt_best_resume \
  --export=ALL,WANDB_RUN_NAME=dwnt_r10_best_resume_20260218_130152,SAVE_DIR=/gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM/outputs/runs/md22_dwnt/local_tmux/dwnt_local_r10_state_resume_20260218_013331,TOTAL_NUM_STEPS=160000,SAVE_EPOCH_INTERVAL=100,VAL_EPOCH_INTERVAL=0,VAL_BATCH_INTERVAL=0,TEST_BATCH_INTERVAL=20000,MAX_LR=5e-5,MIN_LR=5e-6,WARMUP_STEPS=2000,SEED=59,MAX_RADIUS=10.0,PBC_MAX_RADIUS=10.0,MAX_NEIGHBORS=24,FMM_NUM_KAPPA=4,FMM_KAPPA_MIN=0.8,FMM_KAPPA_MAX=1.2,FMM_NUM_DIRECTIONS=16,FMM_COMPUTE_DTYPE=bf16,FMM_KAPPA_CHUNK_SIZE=0,FMM_VALUE_HEAD_DIM=8,FMM_LEARNABLE_RADIAL_COEFFS=true,FMM_RADIAL_COEFFS_MODE=per_l_head,FMM_RADIAL_INIT_SCALE=0.05,FMM_RADIAL_LOW_KAPPA_BIAS=2.0,FMM_COUPLING_NORM=count,HYBRID_LONG_SCALE_INIT=1.0 \
  scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch
```

## Update (2026-02-18 13:06 EST) - Local tmux stateful resume launched from best run

Local resume launch:
- Session: `dwnt_local_best_resume_20260218_130359`
- W&B run name: `dwnt_local_best_resume_20260218_130359`
- Runtime script: `/tmp/dwnt_local_best_resume_20260218_130359.sh`
- Log: `outputs/local_tmux/dwnt_local_best_resume_20260218_130359.log`
- Save dir reused (true stateful): `outputs/runs/md22_dwnt/local_tmux/dwnt_local_r10_state_resume_20260218_013331`

Startup verification:
- tmux session running and GPU process attached (`~55.8 GiB` on `CUDA_VISIBLE_DEVICES=0`).
- Log confirms:
  - `Resume from checkpoint: .../checkpoint_E1500.pt`
  - `optimizer is loaded from checkpoint checkpoint_E1500.pt`
  - `Start Training for epoch: 1500`
  - first resumed train log at epoch 1500 and checkpoint rewrite `checkpoint_E1500.pt`.

## Update (2026-02-18 13:54 EST) - Hybrid Muon optimizer path (hidden 2D only)

Implemented optimizer option:
- Added `optimizer_name` (`adamw|muon`) in run config.
- Added Muon hyperparameters:
  - `muon_beta` (default `0.95`)
  - `muon_ns_steps` (default `5`)
  - `muon_ns_eps` (default `1e-7`)
  - `muon_nesterov` (default `True`)
- Added `MuonAdamW` optimizer in `src/molfm/models/molfm_optimizer.py`.
  - `optim_type='muon'` groups use Newton-Schulz orthogonalized updates.
  - `optim_type='adamw'` groups use AdamW updates.

Hidden-layer selection criterion used for Muon:
- Parameter must satisfy all:
  - `requires_grad=True`
  - tensor rank `ndim == 2`
  - name suffix `*.weight`
  - name prefix `decoder.decoder.blocks.`

Parameter identification snapshots:
- Baseline first-order config (`backbone_config=e2former`): `120` Muon-selected params.
- MD22 serial-hybrid config (`first-order6+fmm-node2`, `nk=4`, `v_head_dim=8`): `104` Muon-selected params.
  - Blocks `0..5`: `14` selected GA weights/block + `1` FFN gating weight/block.
  - Blocks `6..7` (FMM blocks): `6` selected GA weights/block + `1` FFN gating weight/block.

Reproduction / smoke checks:
```bash
PYTHONPATH=./src python -m compileall \
  src/molfm/models/molfm_optimizer.py \
  src/molfm/tasks/train_molfm.py \
  src/molfm/pipeline/schema.py
```

```bash
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM/.conda/envs/e2former-cueq
DS_ACCELERATOR=cpu PYTHONPATH=./src python /tmp/smoke_muon_optimizer.py
```

## Update (2026-02-18 14:10 EST) - Muon support for flattened TP kernels (optional)

Added optional Muon-on-TP path without changing defaults:
- New config knob: `muon_use_tp_flattened` (default `False`).
- When enabled, flattened TP kernel weights are included in Muon groups if they are hidden-layer
  `decoder.decoder.blocks.*.weight` parameters and expose TP instruction metadata.
- `MuonAdamW` now accepts `muon_tp_block_layouts` and applies Muon block-wise:
  each weighted TP path is reshaped to a 2D matrix `(prod(path_shape[:-1]), path_shape[-1])`,
  orthogonalized via Newton-Schulz, then written back to the flattened vector.

Validation snapshot on serial-hybrid (`first-order6+fmm-node2`, `tp_type=QK_alpha+tp_cueq`):
- TP-name parameters: `72`
- TP-name params matched by Muon rule: `18` (these are `rad_func_intputhead.net.*.weight`, 2D)
- True TP core params inspected:
  - `...first_order_tp.tensor_product_tp_component_1.weight` shape `(98304,)` (1D)
  - `...wigner_6j_tp.weight` shape `(1,)` (1D)
  - these are now eligible only when `muon_use_tp_flattened=True`.

## Update (2026-02-18 14:21 EST) - TP Muon validation + bugfix

Ran runtime validation of the new `muon_use_tp_flattened` path on serial-hybrid config.

Findings:
- Initial test exposed a bug: some TP-like modules expose `instructions` but override `.weight` with a scalar proxy (`shape=(1,)`), causing invalid block slicing.
- Fixed in `train_molfm._collect_muon_tp_block_layouts`: now only accepts flattened TP layouts when `sum(weighted_path_numel) == weight.numel()`.

Post-fix checks:
- Grouping counts:
  - without flattened TP: `muon_param_count=104`, `muon_tp_param_count=0`
  - with flattened TP: `muon_param_count=110`, `muon_tp_param_count=6`
- Verified real Muon step on flattened TP weight
  (`decoder.decoder.blocks.0.ga.first_order_tp.tensor_product_tp_component_1.weight`, shape `(98304,)`):
  - finite outputs: True
  - parameter changed after step (`delta=0.6159`)
- Final script result: `RESULT: PASS`.

## Update (2026-02-18 15:07 EST) - A/B comparison for `muon_use_tp_flattened`

Constraint:
- Local node currently has a single H100 (`GPU 0`) fully occupied by active run
  `dwnt_local_best_resume_20260218_130359`, so no fair GPU A/B was possible without interruption.

Executed controlled CPU optimizer-level A/B (same initialized model, identical random gradients):
- Config base: `e2former_hybrid`, `attn_type=first-order6+fmm-node2`, `tp_type=QK_alpha+tp_cueq`, `fmm_num_kappa=4`, `fmm_value_head_dim=8`.
- Variant A: `muon_use_tp_flattened=False`
- Variant B: `muon_use_tp_flattened=True`
- Each variant: 6 optimizer steps with matched synthetic gradients.

Results:
- Muon param counts:
  - A: `muon_param_count=104`, `muon_tp_param_count=0`
  - B: `muon_param_count=110`, `muon_tp_param_count=6` (one flattened TP weight per local block 0..5)
- Mean step time (CPU, optimizer step only):
  - A: `0.754 s`
  - B: `2.244 s`
  - ratio B/A: `2.97x`
- Example TP parameter update (`decoder.decoder.blocks.0.ga.first_order_tp.tensor_product_tp_component_1.weight`):
  - A delta L2: `0.0905` (updated via AdamW path)
  - B delta L2: `0.00982` (updated via Muon block-wise TP path)

Interpretation:
- Enabling flattened TP Muon works and is active for 6 TP weights.
- It adds noticeable optimizer compute overhead on CPU in this synthetic-step test.

## Update (2026-02-18 15:10 EST) - Full Slurm A/B launched for Muon TP

Goal:
- Run a full MD22 DWNT serial-hybrid Slurm comparison with identical settings except
  `muon_use_tp_flattened`.

Launcher changes:
- Updated `scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch` to accept:
  - `OPTIMIZER_NAME`
  - `MUON_USE_TP_FLATTENED`
  - `MUON_BETA`, `MUON_NS_STEPS`, `MUON_NS_EPS`, `MUON_NESTEROV`
- Wired these into Hydra overrides (`optimizer_name`, `muon_*`).

Submitted jobs (timestamp tag `20260218_150929`):
- `1167209` job name `dwnt_muon_notp_full`
  - W&B run: `dwnt_muon_notp_full_20260218_150929`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=false`
- `1167210` job name `dwnt_muon_tp_full`
  - W&B run: `dwnt_muon_tp_full_20260218_150929`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=true`

Common key overrides:
- `OPTIMIZER_NAME=muon`
- Serial hybrid: `attn_type=first-order6+fmm-node2`, `tp_type=QK_alpha+tp_cueq`
- Cutoff: `MAX_RADIUS=10.0`, `PBC_MAX_RADIUS=10.0`, `MAX_NEIGHBORS=24`
- FMM: `nk=4`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`, `v_head_dim=8`,
  learnable radial on, `FMM_COUPLING_NORM=count`
- Optimization: `MAX_LR=5e-5`, `MIN_LR=5e-6`, `WARMUP_STEPS=2000`, `WEIGHT_DECAY=5e-3`, `SEED=59`
- Full-run schedule: defaults `TOTAL_NUM_STEPS=200000`, `TOTAL_NUM_EPOCHS=3000`
- Requested resources per job: `gpu:1`, `cpus=16`, `mem=120G`, `time=47:00:00`
- Batch consistency via accumulation: `PER_GPU_BATCH=2`, `GRAD_ACCUM=4` (global train batch stays 8)

Current scheduler state right after submit:
- `1167209`: `PENDING (Priority)`
- `1167210`: `PENDING (Priority)`
- (No `QOSMaxGRESPerUser` blocker for these two at submit time.)

Expected logs/artifacts:
- Slurm logs:
  - `outputs/slurm/dwnt_muon_notp_full-1167209.out`
  - `outputs/slurm/dwnt_muon_tp_full-1167210.out`
- Save dirs:
  - `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_muon_notp_full_20260218_150929`
  - `outputs/runs/md22_dwnt/hybrid_serial_cueq/dwnt_muon_tp_full_20260218_150929`

## Update (2026-02-18 16:41 EST) - Re-launch Muon A/B with `GRAD_ACCUM=1`

Reason:
- Prior Muon A/B (`1167209`, `1167210`) was launched with `GRAD_ACCUM=4` to keep
  larger effective batch. For direct optimizer comparison against Adam-pure runs,
  we switched Muon runs to `GRAD_ACCUM=1`.

Actions:
- Cancelled previous Muon jobs:
  - `1167209` (`dwnt_muon_notp_full`) -> `CANCELLED`
  - `1167210` (`dwnt_muon_tp_full`) -> `CANCELLED`
- Re-submitted Muon A/B with the same architecture/hyperparameters but `GRAD_ACCUM=1`.

Submitted jobs (timestamp tag `20260218_164118`):
- `1167522` job name `dwnt_muon_notp_ga1`
  - W&B run: `dwnt_muon_notp_ga1_20260218_164118`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=false`
- `1167523` job name `dwnt_muon_tp_ga1`
  - W&B run: `dwnt_muon_tp_ga1_20260218_164118`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=true`

Key overrides for this relaunch:
- `OPTIMIZER_NAME=muon`
- Serial hybrid: `attn_type=first-order6+fmm-node2`, `tp_type=QK_alpha+tp_cueq`
- Cutoff: `MAX_RADIUS=10.0`, `PBC_MAX_RADIUS=10.0`, `MAX_NEIGHBORS=24`
- FMM: `nk=4`, `kappa=[0.8,1.2]`, `dirs=16`, `dtype=bf16`, `v_head_dim=8`,
  learnable radial on, `FMM_COUPLING_NORM=count`
- Optimization: `MAX_LR=5e-5`, `MIN_LR=5e-6`, `WARMUP_STEPS=2000`,
  `WEIGHT_DECAY=5e-3`, `SEED=59`
- Requested resources per job: `gpu:1`, `cpus=16`, `mem=120G`, `time=47:00:00`

## Update (2026-02-18 16:46 EST) - Added Muon A/B with `PER_GPU_BATCH=8`, `GRAD_ACCUM=1`

Per request, submitted an additional pair with local micro-batch increased to 8
while keeping accumulation at 1.

Submitted jobs (timestamp tag `20260218_164622`):
- `1167534` job name `dwnt_muon_notp_b8ga1`
  - W&B run: `dwnt_muon_notp_b8ga1_20260218_164622`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=false`
- `1167535` job name `dwnt_muon_tp_b8ga1`
  - W&B run: `dwnt_muon_tp_b8ga1_20260218_164622`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=true`

Key batch setting for this pair:
- `PER_GPU_BATCH=8`
- `GRAD_ACCUM=1`
- single GPU (`gres/gpu:1`) so effective global train batch is 8.

Scheduler snapshot right after submit:
- `1167534`: `PENDING (Priority)`
- `1167535`: `PENDING (Priority)`

## Update (2026-02-18 16:48 EST) - Removed `PER_GPU_BATCH=2` Muon pair

Per request, removed the `ga1` pair from queue:
- `1167522` (`dwnt_muon_notp_ga1`) -> `CANCELLED`
- `1167523` (`dwnt_muon_tp_ga1`) -> `CANCELLED`

Active Muon pair now:
- `1167534` (`dwnt_muon_notp_b8ga1`)
- `1167535` (`dwnt_muon_tp_b8ga1`)

## Update (2026-02-19 12:03 EST) - `b8ga1` failure diagnosis and `b4ga2` relaunch

Failure diagnosis for `PER_GPU_BATCH=8`, `GRAD_ACCUM=1` pair:
- `1167534` (`dwnt_muon_notp_b8ga1`) -> `FAILED (ExitCode=1:0)`
- `1167535` (`dwnt_muon_tp_b8ga1`) -> `FAILED (ExitCode=1:0)`
- Root cause in both `.err` logs: `torch.OutOfMemoryError` during force
  autograd path (`heads.py`), with attempted allocation `~938 MiB` on a 44.4 GiB GPU
  when only `~703 MiB` free.

User-requested relaunch with reduced micro-batch and increased accumulation:
- `PER_GPU_BATCH=4`
- `GRAD_ACCUM=2`
- Effective global train batch remains `8` on single-GPU jobs.

Submitted jobs (timestamp tag `20260219_120351`):
- `1171450` job name `dwnt_muon_notp_b4ga2`
  - W&B run: `dwnt_muon_notp_b4ga2_20260219_120351`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=false`
- `1171451` job name `dwnt_muon_tp_b4ga2`
  - W&B run: `dwnt_muon_tp_b4ga2_20260219_120351`
  - Muon TP setting: `MUON_USE_TP_FLATTENED=true`

Scheduler snapshot after submit:
- `1171450`: `PENDING (Priority)`
- `1171451`: `PENDING (Priority)`

## Update (2026-02-19 12:24 EST) - Storage cleanup (checkpoint pruning)

Performed targeted cleanup to free space while preserving active training runs.

Kept active run dirs untouched:
- `dwnt_r10_fs_lr50_20260218_013050`
- `dwnt_r10_fs_lr55_20260218_013050`
- `dwnt_muon_notp_b4ga2_20260219_120351`
- `dwnt_muon_tp_b4ga2_20260219_120351`

Pruned old heavy serial-hybrid run dirs by keeping only latest `checkpoint_E*.pt`
and rewriting `checkpoint_list.txt`:
- `dwnt_s6g2_count_fp32_20260215_121259` (kept `checkpoint_E900.pt`)
- `dwnt_s6g2_count_r10_20260215_121259` (kept `checkpoint_E900.pt`)
- `dwnt_s6g2_count_radboost_20260215_121259` (kept `checkpoint_E900.pt`)
- `dwnt_s5g3_count_20260215_121259` (kept `checkpoint_E1050.pt`)
- `dwnt_s6g2_count_ctrl_20260215_121259` (kept `checkpoint_E1050.pt`)

Checkpoint files removed:
- `96` files total (`18+18+18+21+21`)

Space impact:
- Those 5 dirs: ~`6.2G` -> ~`0.30G`
- `outputs/runs` total: ~`11G` -> ~`4.7G`
- Reclaimed: ~`6.3G`

## Update (2026-02-22) - Run analysis snapshot (Adam vs Muon)

Scope analyzed:
- Adam baseline family:
  - `1166125` (`dwnt_r10_fs_lr45`)
  - `1166126` (`dwnt_r10_fs_lr50`)
  - `1166127` (`dwnt_r10_fs_lr55`)
- Muon family (`PER_GPU_BATCH=4`, `GRAD_ACCUM=2`):
  - `1171450` (`dwnt_muon_notp_b4ga2`)
  - `1171451` (`dwnt_muon_tp_b4ga2`)

Normalization:
- Adam runs used `world_size=4` while Muon runs used `world_size=1`.
- Used corrected sample axis `corr_samples = total_samples * world_size`.
- Common horizon selected as minimum terminal corrected samples across compared runs:
  - `corr_samples = 640000`.

At common horizon (`corr_samples=640000`), best-so-far train loss:
- `muon_tp_b4ga2`: `0.3835`
- `muon_notp_b4ga2`: `0.3954`
- `adam_lr55`: `0.4521`
- `adam_lr50`: `0.4674`
- `adam_lr45`: `0.4891`

Notes:
- Adam runs were configured with `total_num_steps=80000` and ended `COMPLETED`.
- Muon runs were configured with `total_num_steps=200000` and ended `TIMEOUT`
  at 47h limit:
  - `1171450` last loss `0.3250` at `step=120650`
  - `1171451` last loss `0.3004` at `step=136515`

Related failure taxonomy (recent history):
- `b8ga1` Muon (`1167534`, `1167535`) failed by CUDA OOM.
- Older `s6g2_*` failures on 2026-02-15 (`1160080`-`1160083`) were checkpoint
  write/storage failures (`PytorchStreamWriter failed writing file`).
- `1160102`-`1160104` ended due Slurm time limit.
