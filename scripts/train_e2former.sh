#!/usr/bin/env bash
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'


[ -z "${layers}" ] && layers=8
[ -z "${num_head}" ] && num_head=8
[ -z "${loss_unit}" ] && loss_unit="kcal/mol"
[ -z "${max_lr}" ] && max_lr=1.5e-4
[ -z "${min_lr}" ] && min_lr=0.75e-4
[ -z "${total_num_steps}" ] && total_num_steps=200000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=5000
[ -z "${train_batch_size}" ] && train_batch_size=512
[ -z "${val_batch_size}" ] && val_batch_size=512
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${val_batch_interval}" ] && val_batch_interval=0

[ -z "${strategy}" ] && strategy=DDP
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2500
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=3000

[ -z "${shuffle}" ] && shuffle=True

[ -z "${data_path}" ] && data_path=''
[ -z "${data_path_list}" ] && data_path_list=""
[ -z "${data_path_list_valid}" ] && data_path_list_valid=""
[ -z "${dataset_name_list}" ] && dataset_name_list="deshaw_di_mol"
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="1"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${loadcheck_path}" ] && loadcheck_path=''
[ -z "${save_dir}" ] && save_dir=''
[ -z "${dataset_name}" ] && dataset_name="."

[ -z "${ifresume}" ] && ifresume=True
[ -z "${wandb_group}" ] && wandb_group=test
[ -z "${wandb_team}" ] && wandb_team=test
[ -z "${wandb_project}" ] && wandb_project=test
[ -z "${wandb_key}" ] && wandb_key=test
[ -z "${wandb_run_name}" ] && wandb_run_name=test

# swanlab login -k "Yor Swanlab KEY"

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62352
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1


echo -e "\n\n"
[ -z "${n_gpu}" ] && n_gpu=1
echo "n_gpu: ${n_gpu}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "LOCAL_RANK : ${LOCAL_RANK}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"
echo "OMPI_COMM_WORLD_LOCAL_RANK: ${OMPI_COMM_WORLD_LOCAL_RANK}"

echo -e "\n\n"
echo "n_layers: ${layers}"
echo "num_head: ${num_head}"
echo "max_lr: ${max_lr}"
echo "max_lr: ${min_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "dropout: ${dropout}"
echo "weight_decay: ${weight_decay}"
echo "data_path: ${data_path}"
echo "output_path: ${output_path}"
echo "dataset_name: ${dataset_name}"

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --master_port $MASTER_PORT \
                      --rdzv_conf timeout=24000"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR \
                      --rdzv_conf timeout=24000"
  fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

export use_charge_embedding=True

export save_batch_interval=250000
export train_batch_size=192
export val_batch_size=4
export gradient_accumulation_steps=1
export val_batch_interval=0

export total_num_steps=500000
export warmup_num_steps=10000
export max_lr=1.5e-4
export min_lr=0.75e-4

export strategy=DDP
export use_unified_batch_sampler=True

export save_epoch_interval=1
export molfm_finetune_mode=True

torchrun $DISTRIBUTED_ARGS molfm/tasks/train_molfm.py \
          --config-name=config_molfm.yaml \
          save_dir=$save_dir \
          ifresume=$ifresume \
          loadcheck_path=$loadcheck_path \
          inference_mode=True \
          molfm_finetune_mode=True \
          molfm_finetune_skip_ori_head=True \
          AutoGradForce=True \
          head_module="md_energy_force_multi_head" \
          use_charge_embedding=True \
          loss_fn='mae' \
          max_lr=$max_lr \
          min_lr=$min_lr \
          warmup_num_steps=$warmup_num_steps \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          wandb_run_name=$wandb_run_name \
          gradient_clipping=1.0 \
          energy_loss_weight=0.1 \
          force_loss_weight=0.9 \
          backbone=e2former \
          backbone_config=e2former \
          backbone_config.with_cluster=True \
          backbone_config.use_rdkit=True \
          backbone_config.num_layers=4 \
          backbone_config.irreps_node_embedding="256x0e+256x1e+256x2e" \
          backbone_config.irreps_head="16x0e+16x1e+16x2e" \
          backbone_config.attn_scalar_head=16 \
          backbone_config.num_attn_heads=16 \
          backbone_config.number_of_basis=256 \
          backbone_config.pbc_max_radius=5 \
          backbone_config.max_radius=5 \
          backbone_config.max_neighbors=640 \
          backbone_config.alpha_drop=0.0 \
          backbone_config.drop_path_rate=0 \
          backbone_config.basis_type='gaussiansmear' \
          backbone_config.norm_layer='rms_norm_sh' \
          backbone_config.attn_type='first-order' \
          backbone_config.tp_type='QK_alpha+use_smooth_softmax' \
          backbone_config.edge_embedtype='default' \
          backbone_config.ffn_type='s3' \
          backbone_config.sparse_attn=True \
          backbone_config.encoder='none' \
          backbone_config.encoder_embed_dim=384 \
          backbone_config.ffn_embedding_dim=768 \
          backbone_config.num_attention_heads=8 \
          backbone_config.dropout=0.1 \
          backbone_config.num_encoder_layers=12 \
          loss_unit=$loss_unit \
          weight_decay=$weight_decay \
          data_path=$data_path \
          share_attention_bias=True \
          data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
          data_path_list_valid=\"$data_path_list_valid\" \
          dataset_split_raito=\"$dataset_split_raito\" \
          shuffle=$shuffle \
          seed=12345 \
          strategy=$strategy \
          total_num_steps=$total_num_steps \
          train_batch_size=$train_batch_size val_batch_size=$val_batch_size \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\"  \
          use_unified_batch_sampler=True \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
          save_batch_interval=$save_batch_interval log_interval=$log_interval 
