# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class ParallelStrategy(str, Enum):
    """
    Args:
        DDP: Distributed Data Parallel (DDP) training, where the model is replicated across multiple devices and gradients are synchronized.
    """
    DDP = "DDP"
    Pipeline = "Pipeline"
    ThreeD = "ThreeD"


@dataclass
class RunConfig:
    # training parameters
    seed: int = 46
    total_num_steps: int = 1000
    total_num_epochs: int = 100
    train_batch_size: int = 1
    val_batch_size: int = 1
    val_batch_interval: int = 0
    val_batch_log_interval: int = 1000
    val_batch_log_all_metric: bool = False
    val_epoch_interval: int = 1
    use_unified_batch_sampler: bool = False
    gradient_clipping: float = 1.0

    # parallelisation parameters
    strategy: ParallelStrategy = ParallelStrategy.DDP
    cpu: bool = False

    # IO parameters
    save_dir: str = "./checkpoints"
    save_batch_interval: int = 0
    save_epoch_interval: int = 1
    log_interval: int = 10000
    finetune_from_checkpoint_dir: Optional[str] = None
    finetune_from_checkpoint_id: Optional[str] = None
    ifresume: bool = False
    load_ckpt: bool = False

    # optimizer hyperparameters
    max_lr: float = 0.0001
    init_lr: float = 8e-5
    min_lr: float = 8e-6
    weight_decay: float = 0.0
    beta1: float = 0.9  # Adam
    beta2: float = 0.999  # Adam
    eps: float = 1e-8  # Adam

    # lr scheduler hyperparameters
    warmup_num_steps: int = 60
    warmup_factor: float = 0.06
    warmup_lr: float = 1e-6
    warmup_num_epochs: int = 10

    # performance parameters
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    auto_cast: bool = False
    grad_scaler_init: float = 1.0
    bf16: bool = False
    fp8: bool = False
    mm_tensorcore: str = "fp32"  # reduce matrix multiplication precision

    compile: bool = False  # compile CUDA kernels with torch.compile

    find_unused_parameters: bool = True

    # dataloader strategy
    ifstack: bool = False
    use_dali_pipeline: bool = False
    unified_data_num_workers: int = 0

    # wandb
    wandb: bool = False
    wandb_team: str = ""
    wandb_group: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""

    # early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "valid_loss"
    early_stopping_mode: str = "min"

    # validate
    calculate_metrics: bool = False

    # profiler
    profiling: bool = False
    prof_dir: str = "./prof"
    ptensorboard: bool = False
    allreduce_log_path: str = "/tmp/stragglers"

    # debugger
    debug: bool = False

    # AutoGradForce
    AutoGradForce: bool = False

    def __str__(self):
        return (
            "Config[\n"
            + "\n".join([f"  {k}: {v}" for k, v in asdict(self).items()])
            + "\n]"
        )


@dataclass
class DistributedConfig(RunConfig):
    """
    DistributedConfig is a combination of RunConfig and distributed settings.

    Args:
        local_rank (int, default: -1): The local rank of the process within a node.

        world_size (int, default: 1): The total number of processes involved in the distributed training.

        node_rank (int, default: 0): The rank of the current node within the cluster.

        rank (int, default: 0): The global rank of the process.

        pipeline_model_parallel_size (int, default: 0): The size of the pipeline model parallel group.

        tensor_model_parallel_size (int, default: 1): The size of the tensor model parallel group.

        deepspeed_config (str, default: ''): The path to the DeepSpeed configuration file.

        dist_backend (str, default: 'nccl'): The distributed backend to use for communication among processes.
    """

    local_rank: int = -1
    world_size: int = 1
    node_rank: int = 0
    rank: int = 0
    pipeline_model_parallel_size: int = 0
    tensor_model_parallel_size: int = 1
    deepspeed_config_path: str = ""
    deepspeed_config: Any = None
    dist_backend: str = "nccl"


@dataclass
class RunState:
    """
    The RunState class helps manage various training-related attributes, making it easier to monitor and control the progress of the training.

    Args:
        args: A RunConfig object that stores the configuration settings for the training process.

        global_step (int, default: 0): The current global step of the training process, which is a count of the number of gradient updates performed.

        epoch (int, default: 0): The current epoch of the training process, representing the number of times the entire dataset has been processed.

        batch (int, default: 0): The current batch number within the current epoch.
    """

    args: RunConfig
    global_step: int = 0
    epoch: int = 0
    batch: int = 0
    sample: int = 0


@dataclass
class StepOutput:
    loss: torch.Tensor
    num_examples: Optional[int] = None
    log_output: Optional[Dict] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


def format_extra_output(raw_extra_output):
    if raw_extra_output is None:
        return ""

    extra_output = []
    for k, v in raw_extra_output.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
                extra_output.append(f"{k}: {v:.4g}")
            else:
                v = v.detach().cpu().numpy()
                extra_output.append(f"{k}: {v}")
        elif isinstance(v, float):
            extra_output.append(f"{k}: {v:.4g}")
        else:
            extra_output.append(f"{k}: {v}")
    extra_output = " | ".join(extra_output)
    return extra_output


@dataclass
class TrainLog:
    loss: float
    grad_scale: float
    lr: float
    epoch: int
    batch: int
    global_step: int
    total_samples: int
    extra_output: Dict

    def __str__(self) -> str:
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Step: {self.global_step} (Epoch {self.epoch} Iter {self.batch+1}) | Loss: {self.loss:.4g} | LR: {self.lr:.4g} | Grad Scale: {self.grad_scale:.4g} | "
            + extra_output
        )


@dataclass
class ValidationLog:
    valid_loss: float
    epoch: int
    num_examples: Optional[int] = None
    extra_output: Optional[Dict] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None

    def __str__(self):
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Valid Loss: {self.valid_loss:.4g} | Num Examples: {self.num_examples} | "
            + extra_output
        )
