# -*- coding: utf-8 -*-
from typing import List, Tuple

import math

from deepspeed.runtime.lr_schedules import WarmupLR
from molfm.logging.loggers import logger

try:
    from apex.optimizers import FusedAdam as _Adam  # isort:skip

    logger.info("apex is installed, using FusedAdam with fp16 optimizer states")

    def _adamw_impl(*args, **kwargs):
        return _Adam(*args, **kwargs, adam_w_mode=True)

except Exception:
    from torch.optim import Adam as _Adam
    from torch.optim import AdamW as _AdamW

    logger.info("apex is not installed, using pytorch AdamW with fp32 optimizer states")

    def _adamw_impl(*args, **kwargs):
        return _AdamW(*args, **kwargs)


WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"
DECAY_LINEAR_RATE = "linear"
DECAY_COSINE_RATE = "cosine"


def parse_name_list(param_list: str = None) -> List[str]:
    if not param_list:
        return []
    items = []
    for name in param_list.strip().split(","):
        name = name.strip()
        if name:
            items.append(name)
    return items


def split_names_and_layers(name_list: List[str]) -> Tuple[List[str], List[int]]:
    names = []
    layers = []
    for name in name_list:
        if isinstance(name, str):
            names.append(name)
        elif isinstance(name, int):
            layers.append(name)
        else:
            raise ValueError(f"Invalid name type: {type(name)}")
    return names, layers


def build_param_groups(net, freeze_list=None, unfreeze_list=None, lr: float = 1e-5, **kwargs):
    freeze_list = parse_name_list(freeze_list)
    unfreeze_list = parse_name_list(unfreeze_list)
    if len(freeze_list) > 0 and len(unfreeze_list) > 0:
        raise ValueError(
            f"freeze_list and unfreeze_list cannot be set at the same time, got {freeze_list}, {unfreeze_list}"
        )

    groups = [{"lr": lr, "params": []}]
    if len(unfreeze_list) > 0 and "dummy" not in unfreeze_list:
        unfreeze_list.append("dummy")

    if len(unfreeze_list) > 0:
        names, layers = split_names_and_layers(unfreeze_list)
        for name, param in net.named_parameters():
            layer_id = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
            if layer_id in layers or any(tag in name for tag in names):
                groups[0]["params"].append(param)
    elif len(freeze_list) > 0:
        names, layers = split_names_and_layers(freeze_list)
        for name, param in net.named_parameters():
            layer_id = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
            if layer_id in layers or any(tag in name for tag in names):
                continue
            groups[0]["params"].append(param)
    else:
        for _, param in net.named_parameters():
            groups[0]["params"].append(param)

    for group in groups:
        group.setdefault("lr", kwargs.get("lr", lr))
        group.setdefault("weight_decay", kwargs.get("weight_decay", 0.0))
    return groups


def make_adam(net, freeze_list=None, unfreeze_list=None, **kwargs):
    param_groups = build_param_groups(
        net, freeze_list=freeze_list, unfreeze_list=unfreeze_list, **kwargs
    )
    return _Adam(param_groups, **kwargs), param_groups


def make_adamw(net, freeze_list=None, unfreeze_list=None, **kwargs):
    param_groups = build_param_groups(
        net, freeze_list=freeze_list, unfreeze_list=unfreeze_list, **kwargs
    )
    return _adamw_impl(param_groups, **kwargs), param_groups


class WarmupDecaySchedule(WarmupLR):
    """Warmup then decay learning rate with optional cosine or linear schedule."""

    def __init__(
        self,
        optimizer,
        total_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LINEAR_RATE,
        last_batch_iteration: int = -1,
        scale_factor: float = 1.0,
        decay_type: str = DECAY_COSINE_RATE,
    ):
        self.total_num_steps = total_num_steps
        super(WarmupDecaySchedule, self).__init__(
            optimizer,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )
        self.scale_factor = scale_factor
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
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        if self.scale_factor != 1.0 and len(self.optimizer.param_groups) > 0:
            for idx, group in enumerate(self.optimizer.param_groups):
                if idx == 0:
                    group["lr"] *= self.scale_factor

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(
                    self.last_batch_iteration + 1
                )
            if self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        if self.decay_type == DECAY_LINEAR_RATE:
            return max(
                0.0,
                float(self.total_num_steps - self.last_batch_iteration)
                / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
            )
        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * float(self.last_batch_iteration - self.warmup_num_steps)
                / float(max(1.0, self.total_num_steps - self.warmup_num_steps))
            )
        )
