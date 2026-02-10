#!/usr/bin/env bash
#SBATCH --job-name=dwnt_e2former_fmm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=47:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -eo pipefail

REPO_ROOT="/gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm outputs/runs/md22_dwnt/fmm_cueq

source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
set +u
conda activate /gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM/.conda/envs/e2former-cueq
set -u

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export DS_ACCELERATOR=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export SWANLAB_ENABLE=0

python -c "import cuequivariance_torch as cueq; print('cuequivariance_torch', getattr(cueq, '__version__', 'unknown'))"
python -c "import loguru, deepspeed; print('deps_ok', True)"
python -c "from molfm.models.e2former.fmm_utils import _cueq_ops_available; print('cueq_ops_available', _cueq_ops_available())"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/outputs/md22_full_lmdb/lmdb}"
DATA_FILE="${DATA_FILE:-md22_double_walled_nanotube.lmdb}"
DATASET_NAME="${DATASET_NAME:-md}"

SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/runs/md22_dwnt/fmm_cueq}"
TOTAL_NUM_STEPS="${TOTAL_NUM_STEPS:-200000}"
TOTAL_NUM_EPOCHS="${TOTAL_NUM_EPOCHS:-3000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LR="${MAX_LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-3}"
SEED="${SEED:-48}"
MASTER_PORT="${MASTER_PORT:-29622}"
WANDB_ENABLE="${WANDB_ENABLE:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-ffm}"
WANDB_GROUP="${WANDB_GROUP:-md22_dwnt}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dwnt_e2former_fmm_cueq_${SLURM_JOB_ID:-local}}"
FMM_NUM_KAPPA="${FMM_NUM_KAPPA:-6}"
FMM_KAPPA_MIN="${FMM_KAPPA_MIN:-0.8}"
FMM_KAPPA_MAX="${FMM_KAPPA_MAX:-1.2}"

echo "Running E2Former-FMM(cuEquivariance TP) on MD22 double_walled_nanotube"
echo "Data: ${DATA_DIR}/${DATA_FILE}"
echo "Save dir: ${SAVE_DIR}"

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  src/molfm/tasks/train_molfm.py \
  --config-name=config_molfm.yaml \
  save_dir="${SAVE_DIR}" \
  ifresume=True \
  inference_mode=False \
  molfm_finetune_mode=True \
  molfm_finetune_skip_ori_head=True \
  AutoGradForce=True \
  head_module=md_energy_force_multi_head \
  loss_fn=mae \
  loss_unit=kcal/mol \
  energy_loss_weight=0.2 \
  force_loss_weight=0.8 \
  backbone=e2former \
  backbone_config=e2former_fmm \
  backbone_config.tp_type=fmm-node+tp_cueq \
  backbone_config.fmm_num_kappa="${FMM_NUM_KAPPA}" \
  backbone_config.fmm_kappa_min="${FMM_KAPPA_MIN}" \
  backbone_config.fmm_kappa_max="${FMM_KAPPA_MAX}" \
  max_lr="${MAX_LR}" \
  min_lr="${MIN_LR}" \
  warmup_num_steps="${WARMUP_STEPS}" \
  weight_decay="${WEIGHT_DECAY}" \
  data_path="${DATA_DIR}" \
  data_path_list="${DATA_FILE}" \
  dataset_name_list="${DATASET_NAME}" \
  data_path_list_valid=none \
  dataset_split_raito=1.0 \
  dataset_micro_batch_size=1 \
  shuffle=True \
  use_unified_batch_sampler=False \
  total_num_steps="${TOTAL_NUM_STEPS}" \
  total_num_epochs="${TOTAL_NUM_EPOCHS}" \
  train_batch_size="${TRAIN_BATCH_SIZE}" \
  val_batch_size="${VAL_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRAD_ACCUM}" \
  wandb="${WANDB_ENABLE}" \
  wandb_project="${WANDB_PROJECT}" \
  wandb_group="${WANDB_GROUP}" \
  wandb_run_name="${WANDB_RUN_NAME}" \
  seed="${SEED}" \
  md22_protocol=True \
  md22_molecule=double_walled_nanotube \
  md22_sample_size=-1 \
  md22_train_prop=0.95 \
  md22_seed="${SEED}"
