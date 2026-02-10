# Plan / Working Memory

Last updated: 2026-02-10

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

## Current Slurm Status (as of 2026-02-10)

Running:
- `1153689` `dwnt_serial_v8_bench`: serial hybrid training with `FMM_VALUE_HEAD_DIM=8` (also ran the speed bench above).
- `1153471` `dwnt_serial_d16_bf16_nk4`: serial hybrid variant (nk=4).
- `1153391` `ood-vscode-proxy`: dev proxy.

Held (queued but user-held):
- `1153474` `dwnt_serial_d09_bf16`
- `1153475` `dwnt_serial_d16_bf16_lr1e4`
- `1153531` `dwnt_serial_r7` (smaller local cutoff)
- `1153532` `dwnt_serial_r6` (smaller local cutoff)

Cancelled to free MaxGRESPerAccount quota:
- `1153472` `dwnt_serial_d25_fp32`
- `1153473` `dwnt_serial_d16_bf16`

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

