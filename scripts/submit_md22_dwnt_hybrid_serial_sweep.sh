#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM"
cd "${REPO_ROOT}"

SCRIPT="scripts/slurm_train_md22_dwnt_e2former_hybrid_serial_cueq.sbatch"
TS="$(date +%Y%m%d_%H%M%S)"

submit() {
  local job_name="$1"
  local run_name="$2"
  shift 2

  local job_id
  job_id="$(sbatch -J "${job_name}" --export=ALL,WANDB_RUN_NAME="${run_name}",$* "${SCRIPT}" | awk '{print $NF}')"
  echo "${job_name}: ${job_id} (${run_name})"
}

echo "Submitting DWNT serial-hybrid sweep (DDP multi-GPU via ${SCRIPT})"
echo "Timestamp: ${TS}"
echo

# Keep the serial split fixed (6 local + 2 global) and vary only the global/FMM speed knobs + LR.
# DDP batch size defaults are computed inside the sbatch script via PER_GPU_BATCH and WORLD_SIZE.

submit \
  "dwnt_serial_d25_fp32" \
  "dwnt_serial_l6g2_d25_fp32_nk6_lr5e-5_s55_${TS}" \
  "FMM_NUM_DIRECTIONS=25,FMM_COMPUTE_DTYPE=fp32,FMM_NUM_KAPPA=6,MAX_LR=5e-5,MIN_LR=5e-6,SEED=55"

submit \
  "dwnt_serial_d16_bf16" \
  "dwnt_serial_l6g2_d16_bf16_nk6_lr5e-5_s56_${TS}" \
  "FMM_NUM_DIRECTIONS=16,FMM_COMPUTE_DTYPE=bf16,FMM_NUM_KAPPA=6,MAX_LR=5e-5,MIN_LR=5e-6,SEED=56"

submit \
  "dwnt_serial_d09_bf16" \
  "dwnt_serial_l6g2_d09_bf16_nk6_lr5e-5_s57_${TS}" \
  "FMM_NUM_DIRECTIONS=9,FMM_COMPUTE_DTYPE=bf16,FMM_NUM_KAPPA=6,MAX_LR=5e-5,MIN_LR=5e-6,SEED=57"

submit \
  "dwnt_serial_d16_bf16_lr1e4" \
  "dwnt_serial_l6g2_d16_bf16_nk6_lr1e-4_s58_${TS}" \
  "FMM_NUM_DIRECTIONS=16,FMM_COMPUTE_DTYPE=bf16,FMM_NUM_KAPPA=6,MAX_LR=1e-4,MIN_LR=1e-5,SEED=58"

submit \
  "dwnt_serial_d16_bf16_nk4" \
  "dwnt_serial_l6g2_d16_bf16_nk4_lr5e-5_s59_${TS}" \
  "FMM_NUM_DIRECTIONS=16,FMM_COMPUTE_DTYPE=bf16,FMM_NUM_KAPPA=4,MAX_LR=5e-5,MIN_LR=5e-6,SEED=59"

echo
echo "Track with:"
echo "  squeue -u $(whoami)"

