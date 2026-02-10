#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM"
cd "${REPO_ROOT}"

mkdir -p outputs/slurm

BASELINE_SCRIPT="scripts/slurm_train_md22_dwnt_e2former.sbatch"
FMM_SCRIPT="scripts/slurm_train_md22_dwnt_e2former_fmm_cueq.sbatch"
HYBRID_SCRIPT="scripts/slurm_train_md22_dwnt_e2former_hybrid_cueq.sbatch"

echo "Submitting baseline job: ${BASELINE_SCRIPT}"
BASELINE_JOB="$(sbatch "${BASELINE_SCRIPT}" | awk '{print $NF}')"
echo "baseline job id: ${BASELINE_JOB}"

echo "Submitting FMM cuEquivariance job: ${FMM_SCRIPT}"
FMM_JOB="$(sbatch "${FMM_SCRIPT}" | awk '{print $NF}')"
echo "fmm job id: ${FMM_JOB}"

echo "Submitting Hybrid local+FMM cuEquivariance job: ${HYBRID_SCRIPT}"
HYBRID_JOB="$(sbatch "${HYBRID_SCRIPT}" | awk '{print $NF}')"
echo "hybrid job id: ${HYBRID_JOB}"

echo
echo "Track with:"
echo "  squeue -j ${BASELINE_JOB},${FMM_JOB},${HYBRID_JOB}"
echo "Logs:"
echo "  outputs/slurm/dwnt_e2former-${BASELINE_JOB}.out"
echo "  outputs/slurm/dwnt_e2former_fmm-${FMM_JOB}.out"
echo "  outputs/slurm/dwnt_e2former_hybrid-${HYBRID_JOB}.out"
