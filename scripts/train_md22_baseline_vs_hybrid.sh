#!/usr/bin/env bash
set -euo pipefail

ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${n_gpu:-}" ] && n_gpu=1
[ -z "${MASTER_ADDR:-}" ] && MASTER_ADDR=127.0.0.1
[ -z "${MASTER_PORT:-}" ] && MASTER_PORT=29517

[ -z "${data_path:-}" ] && data_path=""
[ -z "${data_path_list:-}" ] && data_path_list=""
[ -z "${dataset_name_list:-}" ] && dataset_name_list="md"
[ -z "${dataset_split_raito:-}" ] && dataset_split_raito="1.0"
[ -z "${dataset_micro_batch_size:-}" ] && dataset_micro_batch_size="1"

[ -z "${md22_molecule:-}" ] && md22_molecule=""
[ -z "${md22_sample_size:-}" ] && md22_sample_size=-1
[ -z "${md22_train_prop:-}" ] && md22_train_prop=0.95
[ -z "${seed:-}" ] && seed=48

[ -z "${train_batch_size:-}" ] && train_batch_size=32
[ -z "${val_batch_size:-}" ] && val_batch_size=32
[ -z "${gradient_accumulation_steps:-}" ] && gradient_accumulation_steps=1
[ -z "${total_num_steps:-}" ] && total_num_steps=200000
[ -z "${warmup_num_steps:-}" ] && warmup_num_steps=5000
[ -z "${max_lr:-}" ] && max_lr=5e-4
[ -z "${min_lr:-}" ] && min_lr=1e-6
[ -z "${epochs:-}" ] && epochs=3000
[ -z "${energy_loss_weight:-}" ] && energy_loss_weight=0.2
[ -z "${force_loss_weight:-}" ] && force_loss_weight=0.8
[ -z "${weight_decay:-}" ] && weight_decay=5e-3

[ -z "${save_root:-}" ] && save_root="./outputs/md22_protocol"
[ -z "${run_baseline:-}" ] && run_baseline=1
[ -z "${run_hybrid:-}" ] && run_hybrid=1

if [[ -z "${data_path_list}" ]]; then
  echo "data_path_list is empty. Set it to an MD22 LMDB path (relative to data_path or absolute)." >&2
  exit 1
fi

if [[ "${run_baseline}" == "0" && "${run_hybrid}" == "0" ]]; then
  echo "Both run_baseline and run_hybrid are disabled. Nothing to run." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}/src"

DISTRIBUTED_ARGS="--nproc_per_node ${n_gpu} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --rdzv_conf timeout=24000"
if command -v torchrun >/dev/null 2>&1; then
  LAUNCH_CMD=(torchrun)
else
  LAUNCH_CMD=(python -m torch.distributed.run)
fi

run_variant () {
  local variant="$1"
  local backbone_cfg="$2"
  local molecule_tag="${md22_molecule:-auto}"
  local run_dir="${save_root}/${molecule_tag}/${variant}"

  "${LAUNCH_CMD[@]}" ${DISTRIBUTED_ARGS} molfm/tasks/train_molfm.py \
    --config-name=config_molfm.yaml \
    save_dir="${run_dir}" \
    ifresume=False \
    inference_mode=False \
    molfm_finetune_mode=True \
    molfm_finetune_skip_ori_head=True \
    AutoGradForce=True \
    head_module="md_energy_force_multi_head" \
    loss_fn="mae" \
    loss_unit="kcal/mol" \
    energy_loss_weight="${energy_loss_weight}" \
    force_loss_weight="${force_loss_weight}" \
    backbone="e2former" \
    backbone_config="${backbone_cfg}" \
    max_lr="${max_lr}" \
    min_lr="${min_lr}" \
    warmup_num_steps="${warmup_num_steps}" \
    weight_decay="${weight_decay}" \
    data_path="${data_path}" \
    data_path_list="${data_path_list}" \
    dataset_name_list="${dataset_name_list}" \
    data_path_list_valid="none" \
    dataset_split_raito="${dataset_split_raito}" \
    dataset_micro_batch_size="${dataset_micro_batch_size}" \
    shuffle=True \
    use_unified_batch_sampler=False \
    total_num_steps="${total_num_steps}" \
    total_num_epochs="${epochs}" \
    train_batch_size="${train_batch_size}" \
    val_batch_size="${val_batch_size}" \
    gradient_accumulation_steps="${gradient_accumulation_steps}" \
    seed="${seed}" \
    md22_protocol=True \
    md22_molecule="${md22_molecule}" \
    md22_sample_size="${md22_sample_size}" \
    md22_train_prop="${md22_train_prop}" \
    md22_seed="${seed}"
}

if [[ "${run_baseline}" != "0" ]]; then
  run_variant "baseline" "e2former"
fi

if [[ "${run_hybrid}" != "0" ]]; then
  run_variant "baseline_plus_ffm" "e2former_hybrid"
fi
