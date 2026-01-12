# -*- coding: utf-8 -*-
import copy
import os
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Union

import torch
from loguru import logger
import wandb  # isort:skip
import swanlab  # isort:skip

handlers = {}

def is_master_node():
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False

def get_logger():
    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{level}</cyan>]: {message}",
            colorize=True,
            filter=console_log_filter,
            enqueue=True,
        )

    return logger


# Custom function to handle tensor attributes
def dataclass_to_dict(dataclass_obj: Union[dataclass, Dict]) -> Dict:
    if isinstance(dataclass_obj, dict):
        return dataclass_obj
    result = {}
    for field in fields(dataclass_obj):
        value = getattr(dataclass_obj, field.name)
        if isinstance(value, torch.Tensor) and not value.is_leaf:
            result[field.name] = value.clone().detach()
        else:
            result[field.name] = value
    return result


class MetricLogger(object):

    def log(self, metrics, prefix: str = "", global_step: int = None,
            log_wandb: bool = True, log_swanlab: bool = True):
        if not is_master_node():
            return

        if isinstance(metrics, dict):
            log_data = dict(metrics)
        elif is_dataclass(metrics):
            log_data = dataclass_to_dict(metrics)

            if "extra_output" in log_data:
                extra_output = log_data["extra_output"]
                if extra_output is not None:
                    for k, v in extra_output.items():
                        log_data[k] = v
                del log_data["extra_output"]
        else:
            logger.warning(f"MetricLogger: unsupported metrics type={type(metrics)}; will stringify for console")
            log_data = {"message": str(metrics)}

        for k, v in list(log_data.items()):
            if isinstance(v, torch.Tensor):

                if v.numel() == 1:
                    log_data[k] = v.detach().item()
                else:
                    try:
                        log_data[k] = v.detach().cpu().tolist()
                    except Exception:
                        log_data[k] = str(v)

        log_str = ""
        for k, v in log_data.items():
            if v is not None:
                log_str += f" | {k}={v} "
        logger.info(log_str)

        prefixed = log_data
        if prefix:
            prefixed = {f"{prefix}/{k}": v for k, v in log_data.items()}

        if log_wandb:
            try:
                if wandb.run is not None:
                    wandb.log(prefixed, step=global_step)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

        if log_swanlab and swanlab is not None:
            try:
                swanlab.log(prefixed, step=global_step)
            except Exception as e:
                logger.warning(f"SwanLab log failed: {e}")


def console_log_filter(record):
    # For message with level INFO, we only log it on master node
    # For others, we log it on all nodes
    if record["level"].name != "INFO":
        return True

    return is_master_node()
