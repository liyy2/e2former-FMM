#!/usr/bin/env bash
# Serial hybrid schedule: run short-range edge attention first, then long-range node-FMM.
# Example default: first-order6 + fmm-node2 (total 8 layers).
#
# Key overrides are passed via Hydra:
#   backbone_config.attn_type="first-order${LOCAL}+fmm-node${GLOBAL}"
#   backbone_config.tp_type="QK_alpha+tp_cueq"
#
# Speed knobs for the FMM blocks:
#   backbone_config.fmm_num_directions
#   backbone_config.fmm_compute_dtype
#
# Note: reducing num_directions improves speed but can hurt equivariance/accuracy.

#SBATCH --job-name=dwnt_e2former_hybrid_serial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=47:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -eo pipefail

REPO_ROOT="/gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm outputs/runs/md22_dwnt/hybrid_serial_cueq

source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
set +u
conda activate /gpfs/radev/project/gerstein/yl2428/yl2428/e2former-FMM/.conda/envs/e2former-cueq
set -u

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export DS_ACCELERATOR=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export SWANLAB_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

python -c "import cuequivariance_torch as cueq; print('cuequivariance_torch', getattr(cueq, '__version__', 'unknown'))"
python -c "from molfm.models.e2former.fmm_utils import _cueq_ops_available; print('cueq_ops_available', _cueq_ops_available())"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/outputs/md22_full_lmdb/lmdb}"
DATA_FILE="${DATA_FILE:-md22_double_walled_nanotube.lmdb}"
DATASET_NAME="${DATASET_NAME:-md}"

TOTAL_NUM_STEPS="${TOTAL_NUM_STEPS:-200000}"
TOTAL_NUM_EPOCHS="${TOTAL_NUM_EPOCHS:-3000}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LR="${MAX_LR:-5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-3}"
SEED="${SEED:-48}"

NUM_LAYERS="${NUM_LAYERS:-8}"
LOCAL_LAYERS="${LOCAL_LAYERS:-6}"
GLOBAL_LAYERS="${GLOBAL_LAYERS:-$((NUM_LAYERS - LOCAL_LAYERS))}"
SERIAL_ATTN_TYPE="${SERIAL_ATTN_TYPE:-first-order${LOCAL_LAYERS}+fmm-node${GLOBAL_LAYERS}}"

# Local cutoff (applies to first-order/edge attention blocks only).
MAX_RADIUS="${MAX_RADIUS:-15.0}"
PBC_MAX_RADIUS="${PBC_MAX_RADIUS:-${MAX_RADIUS}}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-20}"

FMM_NUM_KAPPA="${FMM_NUM_KAPPA:-6}"
FMM_KAPPA_MIN="${FMM_KAPPA_MIN:-0.8}"
FMM_KAPPA_MAX="${FMM_KAPPA_MAX:-1.2}"
FMM_NUM_DIRECTIONS="${FMM_NUM_DIRECTIONS:-16}"
FMM_COMPUTE_DTYPE="${FMM_COMPUTE_DTYPE:-bf16}" # auto|fp32|bf16|fp16
FMM_KAPPA_CHUNK_SIZE="${FMM_KAPPA_CHUNK_SIZE:-0}"
FMM_VALUE_HEAD_DIM="${FMM_VALUE_HEAD_DIM:-0}" # 0=disable; e.g. 8 or 16 to shrink V_dim

LOADCHECK_PATH="${LOADCHECK_PATH:-}"

# Avoid collisions when multiple jobs share a node.
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
# Prefer SLURM_JOB_GPUS when set (e.g. "0,1,2,3") so the count matches the allocation.
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  IFS=',' read -r -a _gpu_ids <<< "${SLURM_JOB_GPUS}"
  GPUS_PER_NODE="${#_gpu_ids[@]}"
else
  GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
fi
WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"

PER_GPU_BATCH="${PER_GPU_BATCH:-2}"
PER_GPU_VAL_BATCH="${PER_GPU_VAL_BATCH:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((PER_GPU_BATCH * WORLD_SIZE * GRAD_ACCUM))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((PER_GPU_VAL_BATCH * WORLD_SIZE))}"

WANDB_ENABLE="${WANDB_ENABLE:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-ffm}"
WANDB_GROUP="${WANDB_GROUP:-md22_dwnt}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dwnt_e2former_hybrid_serial_cueq_${SLURM_JOB_ID:-local}}"

SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/runs/md22_dwnt/hybrid_serial_cueq/${WANDB_RUN_NAME}}"
mkdir -p "${SAVE_DIR}"

echo "Running E2Former serial hybrid on MD22 double_walled_nanotube"
echo "Data: ${DATA_DIR}/${DATA_FILE}"
echo "Save dir: ${SAVE_DIR}"
echo "Serial schedule: ${SERIAL_ATTN_TYPE} (num_layers=${NUM_LAYERS})"
echo "Local cutoff: r=${MAX_RADIUS} max_neighbors=${MAX_NEIGHBORS}"
echo "FMM: nk=${FMM_NUM_KAPPA} kappa=[${FMM_KAPPA_MIN},${FMM_KAPPA_MAX}] dirs=${FMM_NUM_DIRECTIONS} dtype=${FMM_COMPUTE_DTYPE} v_head_dim=${FMM_VALUE_HEAD_DIM}"
echo "DDP: nnodes=${NNODES} node_rank=${NODE_RANK} gpus_per_node=${GPUS_PER_NODE} world_size=${WORLD_SIZE}"
echo "Batch: per_gpu=${PER_GPU_BATCH} grad_accum=${GRAD_ACCUM} train_batch_size(global)=${TRAIN_BATCH_SIZE}"

python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
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
  backbone_config=e2former_hybrid \
  backbone_config.num_layers="${NUM_LAYERS}" \
  backbone_config.max_radius="${MAX_RADIUS}" \
  backbone_config.pbc_max_radius="${PBC_MAX_RADIUS}" \
  backbone_config.max_neighbors="${MAX_NEIGHBORS}" \
  backbone_config.attn_type="${SERIAL_ATTN_TYPE}" \
  backbone_config.tp_type="QK_alpha+tp_cueq" \
  backbone_config.fmm_num_kappa="${FMM_NUM_KAPPA}" \
  backbone_config.fmm_kappa_min="${FMM_KAPPA_MIN}" \
  backbone_config.fmm_kappa_max="${FMM_KAPPA_MAX}" \
  backbone_config.fmm_num_directions="${FMM_NUM_DIRECTIONS}" \
  backbone_config.fmm_compute_dtype="${FMM_COMPUTE_DTYPE}" \
  backbone_config.fmm_kappa_chunk_size="${FMM_KAPPA_CHUNK_SIZE}" \
  backbone_config.fmm_value_head_dim="${FMM_VALUE_HEAD_DIM}" \
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
  md22_seed="${SEED}" \
  loadcheck_path="${LOADCHECK_PATH}"
