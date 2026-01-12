# -*- coding: utf-8 -*-
import math
from itertools import chain
from typing import Any, Dict, List, Optional

from deepspeed.runtime.lr_schedules import WarmupLR
from torch.optim import Optimizer

from molfm.logging import logger

try:
    from apex.optimizers import FusedAdam

    logger.info("Using apex FusedAdam.")
    USE_APEX_FUSED_ADAM = True
except:
    from torch.optim.adamw import AdamW

    logger.info("Using torch AdamW.")
    USE_APEX_FUSED_ADAM = False

import torch

from molfm.logging.loggers import logger
from molfm.pipeline.schema import ParallelStrategy

from torch.optim import Adam  # isort:skip


def process_param(
    net,
    freeze_list: List = [],
    unfreeze_list: List = [],
    lr: float = 1e-5,
    mfm_lora: bool = False,
    **kwargs,
):
    param_groups = [{}, {}]
    param_groups[0]["lr"] = 0.0
    param_groups[0]["params"] = []
    param_groups[1]["lr"] = 0.0
    param_groups[1]["params"] = []

    for name, param in net.named_parameters():
        if name.find("energy_head") != -1 or name.find("forces_head") != -1:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    return param_groups


def myAdam(
    net,
    impl=Adam,
    freeze_list: List = [],
    unfreeze_list: List = [],
    mfm_lora=False,
    **kwargs,
):
    assert (
        len(freeze_list) == 0 or len(unfreeze_list) == 0
    ), "freeze_list and unfreeze_list cannot be set at the same time"

    new_param_groups = []
    param_groups = process_param(
        net,
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
        mfm_lora=mfm_lora,
        **kwargs,
    )
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs)


WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"
DECAY_LINEAR_RATE = "linear"
DECAY_COSINE_RATE = "cosine"


class groupWarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
    over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_num_steps (int): total number of training steps
        warmup_min_lr (float or list): minimum learning rate. Default: 0
        warmup_max_lr (float or list): maximum learning rate. Default: 0.001
        warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
        warmup_type {'log', 'linear'}: increasing function from min_lr to max_lr during warmup. Default: log
        last_batch_iteration (int): The index of the last batch. Default: -1.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = WarmupDecayLR(optimizer, 1000000)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LINEAR_RATE,
        last_batch_iteration: int = -1,
        d_tilde: float = 1.0,
        decay_type: str = DECAY_LINEAR_RATE,
    ):
        self.total_num_steps = total_num_steps
        super(groupWarmupDecayLR, self).__init__(
            optimizer,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )
        self.d_tilde = d_tilde
        self.decay_type = decay_type

        if self.total_num_steps < self.warmup_num_steps:
            logger.warning(
                "total_num_steps {} is less than warmup_num_steps {}".format(
                    total_num_steps, warmup_num_steps
                )
            )
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def step(self, last_batch_iteration=None):
        """Update the learning rate of each parameter group."""
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self.optimizer.param_groups[1]["lr"] /= self.d_tilde

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(
                    self.last_batch_iteration + 1
                )
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        else:
            if self.decay_type == DECAY_LINEAR_RATE:
                return max(
                    0.0,
                    float(self.total_num_steps - self.last_batch_iteration)
                    / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
                )
            else:
                return 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * float(self.last_batch_iteration - self.warmup_num_steps)
                        / float(max(1.0, self.total_num_steps - self.warmup_num_steps))
                    )
                )


class WarmupDecayLR(groupWarmupDecayLR):
    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


if USE_APEX_FUSED_ADAM:

    class AdamFP16(FusedAdam):
        def __init__(
            self,
            params,
            distributed_strategy: ParallelStrategy,
            lr=1e-3,
            bias_correction=True,
            betas=(0.9, 0.999),
            eps=1e-8,
            adam_w_mode=True,
            weight_decay=0.0,
            amsgrad=False,
            set_grad_none=True,
        ):
            # we need a master copy of fp32 model weights for stability training with FP16
            super().__init__(
                params,
                lr,
                bias_correction,
                betas,
                eps,
                adam_w_mode,
                weight_decay,
                amsgrad,
                set_grad_none,
                capturable=True,
                master_weights=True,
            )
            self.moved_master_param = False
            self.distributed_strategy = distributed_strategy
            if self.distributed_strategy not in [
                ParallelStrategy.DDP,
            ]:
                raise ValueError(
                    f"FP16 with FusedAdam is currently not supported with distributed_strategy {self.distributed_strategy}."
                )

        def step(
            self,
            closure=None,
            grads=None,
            output_params=None,
            scale=None,
            grad_norms=None,
            grad_scaler=None,
        ):
            if not self.moved_master_param:
                for group, group_master in zip(
                    self.param_groups, self.param_groups_master
                ):
                    if len(group["params"]) == 0:
                        continue
                    for i, (p, p_master) in enumerate(
                        zip(group["params"], group_master["params"])
                    ):
                        group_master["params"][i] = p_master.to(device=p.device)
                torch.cuda.empty_cache()
                self.moved_master_param = True

            for group in self.param_groups:
                device = group["params"][0].device
                if isinstance(group["lr"], float):
                    device = group["params"][0].device
                    group["lr"] = torch.tensor(
                        group["lr"], dtype=torch.float32, device=device
                    )
                elif isinstance(group["lr"], torch.Tensor):
                    device = group["params"][0].device
                    group["lr"] = group["lr"].to(device=device)
                if "step" in group and isinstance(group["step"], torch.Tensor):
                    device = group["params"][0].device
                    group["step"] = group["step"].to(device=device)
                self._dummy_overflow_buf = self._dummy_overflow_buf.to(device=device)

            return super().step(
                closure, grads, output_params, scale, grad_norms, grad_scaler
            )

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            super().load_state_dict(state_dict)
            for group, group_master in zip(self.param_groups, self.param_groups_master):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for key in group:
                    if isinstance(group[key], torch.Tensor):
                        group[key] = group[key].to(device=device)
                for i, p in enumerate(group["params"]):
                    group_master["params"][i] = p.clone().detach().float()

            saved_groups = state_dict["param_groups"]
            groups = self.param_groups
            id_map = {
                old_id: p
                for old_id, p in zip(
                    chain.from_iterable((g["params"] for g in saved_groups)),
                    chain.from_iterable((g["params"] for g in groups)),
                )
            }

            for old_id in id_map:
                if old_id in state_dict["state"]:
                    state = state_dict["state"][old_id]
                    fp32_state_names = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
                    for state_name in fp32_state_names:
                        if state_name in state:
                            p = id_map[old_id]
                            self.state[p][state_name] = state[state_name].to(
                                device=p.device
                            )

else:

    class AdamFP16(AdamW):
        def __init__(
            self,
            params,
            distributed_strategy: ParallelStrategy,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False,
            *,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            capturable: bool = False,
            differentiable: bool = False,
            fused: Optional[bool] = None,
        ) -> None:
            super().__init__(
                params,
                lr,
                betas,
                eps,
                weight_decay,
                amsgrad,
                foreach=foreach,
                maximize=maximize,
                capturable=capturable,
                differentiable=differentiable,
                fused=fused,
            )
            self.distributed_strategy = distributed_strategy

        def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        ):
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    # Use FP32 for optimizer states
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=torch.float
                    )
                    # Exponential moving average of squared gradient values
                    # Use FP32 for optimizer states
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=torch.float
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        # Use FP32 for optimizer states
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=torch.float
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            super().load_state_dict(state_dict)

            saved_groups = state_dict["param_groups"]
            groups = self.param_groups
            id_map = {
                old_id: p
                for old_id, p in zip(
                    chain.from_iterable((g["params"] for g in saved_groups)),
                    chain.from_iterable((g["params"] for g in groups)),
                )
            }

            for old_id in id_map:
                if old_id in state_dict["state"]:
                    state = state_dict["state"][old_id]
                    fp32_state_names = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
                    for state_name in fp32_state_names:
                        if state_name in state:
                            p = id_map[old_id]
                            self.state[p][state_name] = state[state_name].to(
                                device=p.device
                            )
