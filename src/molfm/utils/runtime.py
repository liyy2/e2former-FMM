# -*- coding: utf-8 -*-
import ast
import inspect
import os
import typing
from argparse import ArgumentParser
from dataclasses import MISSING, Field, asdict, dataclass, fields, is_dataclass
from enum import Enum
from functools import wraps
from typing import List, Type

import yaml

from molfm.logging import logger
from molfm.pipeline.loop import seed_all

import wandb  # isort:skip
import swanlab  # isort:skip


def is_master_node():
    return "RANK" not in os.environ or int(os.environ["RANK"]) == 0


def set_env(args):
    import torch

    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if os.environ.get("LOCAL_RANK") is not None:
        _apply_distributed_env(args)


def _apply_distributed_env(args):
    import torch

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(args.local_rank)

    logger.success(
        "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
        )
    )


def wandb_init(args):
    if not is_master_node():
        return
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("Wandb not configured, logging to console only")
        return
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name,
        entity=args.wandb_team if args.wandb_team != "" else None,
        config=args,
        settings=wandb.Settings(
            init_timeout=300,
            _disable_stats=True,
        ),
    )


def swanlab_init(args):
    if not is_master_node():
        return
    if os.getenv("SWANLAB_ENABLE", "1") == "0":
        logger.warning("SwanLab disabled via SWANLAB_ENABLE=0")
        return
    if swanlab is None:
        logger.warning("SwanLab not installed, skip SwanLab init")
        return

    _set_env_if_present("SWANLAB_API_HOST")
    _set_env_if_present("SWANLAB_WEB_HOST")

    project = os.getenv(
        "SWANLAB_PROJECT",
        getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "default",
    )
    exp_name = os.getenv(
        "SWANLAB_EXPERIMENT",
        getattr(args, "swanlab_run_name", None) or getattr(args, "wandb_run_name", None),
    )
    workspace = os.getenv(
        "SWANLAB_GROUP",
        getattr(args, "swanlab_group_name", None) or getattr(args, "wandb_group", None),
    )

    cfg_dict = _safe_config_dict(args)
    try:
        swanlab.init(project=project, experiment_name=exp_name, workspace=workspace, config=cfg_dict)
        args.swanlab = True
        logger.success(f"SwanLab inited: project={project}, experiment={exp_name}")
    except Exception as e:
        logger.warning(f"SwanLab init failed: {e}")


def cli(*cfg_classes_and_funcs):
    def decorator(main):
        @wraps(main)
        def wrapper():
            args = _build_argparse_args(cfg_classes_and_funcs)
            logger.info(args)
            seed_all(args.seed)
            set_env(args)
            _maybe_init_wandb(args)

            logger.success(
                "====================================Start!===================================="
            )
            try:
                main(args)
            except Exception as e:
                logger.exception(e)
                logger.error(
                    "====================================Fail!===================================="
                )
                exit()

            logger.success(
                "====================================Done!===================================="
            )

        return wrapper

    return decorator


def hydracli(*cfg_classes_and_funcs, conifg_path):
    def decorator(main):
        @wraps(main)
        def wrapper():
            cfg_classes = []
            for cfg in cfg_classes_and_funcs:
                if inspect.isclass(cfg):
                    cfg_classes.append(cfg)
                else:
                    logger.warning(f"cfg_func {cfg} is not supported in hydracli")

            args = add_dataclass_to_dictconfig(cfg_classes, conifg_path)
            logger.info(args)

            seed_all(args.seed)
            set_env(args)

            if is_master_node():
                _maybe_init_wandb_hydra(args)

            logger.success(
                "====================================Start!===================================="
            )
            try:
                main(args)
            except Exception as e:
                logger.exception(e)
                logger.error(
                    "====================================Fail!===================================="
                )
                exit()

            logger.success(
                "====================================Done!===================================="
            )

        return wrapper

    return decorator


def add_dataclass_to_parser(configs, parser: ArgumentParser):
    for config in configs:
        group = parser.add_argument_group(config.__name__)
        for field in fields(config):
            name = field.name.replace("-", "_")
            field_type = _unwarp_optional(field.type)
            if _argument_exists(parser, name):
                logger.warning(f"Duplicate config name: {name}, not added to parser")
                continue

            default = _resolve_default(field)
            if field_type == bool:
                group.add_argument("--" + name, action="store_true", default=default)
                group.add_argument("--no_" + name, action="store_false", dest=name)
            elif _is_enum_type(field_type):
                parse_enum = _make_enum_parser(field.type)
                group.add_argument("--" + name, type=parse_enum, default=default)
            elif _is_collection(field_type):
                group.add_argument("--" + name, type=ast.literal_eval, default=default)
            else:
                group.add_argument("--" + name, type=field_type, default=default)
    return parser


def add_dataclass_to_dictconfig(configs: List[Type[dataclass]], config_path: str):
    fields_map = {
        field.name: (field.type, Field())
        for config in configs
        for field in fields(config)
    }
    Config = type("Config", (object,), fields_map)
    Config = dataclass(Config)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config(**data)


def _argument_exists(parser, arg_name):
    if arg_name.startswith("--"):
        arg_name = arg_name[2:]
    args_name_pre = "--" + arg_name
    return (
        arg_name in parser._option_string_actions
        or args_name_pre in parser._option_string_actions
    )


def _unwarp_optional(field_type: typing.Type):
    if typing.get_origin(field_type) == typing.Union:
        args = typing.get_args(field_type)
        if len(args) == 2 and args[1] == type(None) and args[0] != type(None):
            return args[0]
    return field_type


def _is_enum_type(tp: typing.Type) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _is_collection(tp: typing.Type) -> bool:
    if hasattr(tp, "__origin__"):
        tp = tp.__origin__
    return tp in (list, tuple, set)


def _make_enum_parser(enum: Enum):
    choices = [e.name for e in enum]

    def parse_enum(arg):
        try:
            return enum[arg]
        except KeyError:
            raise ValueError(
                f"Invalid choice: {arg} for type {enum.__name__}. Valid choices are: {choices}"
            )

    return parse_enum


def _resolve_default(field):
    if field.default != MISSING:
        return field.default
    if field.default_factory != MISSING:
        return field.default_factory()
    return None


def _build_argparse_args(cfg_classes_and_funcs):
    parser = ArgumentParser()
    cfg_classes = []
    cfg_funcs = []
    for cfg in cfg_classes_and_funcs:
        if inspect.isclass(cfg):
            cfg_classes.append(cfg)
        else:
            cfg_funcs.append(cfg)
    for cfg_func in cfg_funcs:
        parser = cfg_func(parser)
    parser = add_dataclass_to_parser(cfg_classes, parser)
    return parser.parse_args()


def _safe_config_dict(args):
    if is_dataclass(args):
        return asdict(args)
    return dict(vars(args)) if hasattr(args, "__dict__") else {}


def _set_env_if_present(key):
    value = os.getenv(key)
    if value:
        os.environ[key] = value


def _maybe_init_wandb(args):
    if not is_master_node():
        return
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("Wandb not configured, logging to console only")
        return

    args.wandb = True
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    wandb_team = os.getenv("WANDB_TEAM")
    wandb_group = os.getenv("WANDB_GROUP")

    args.wandb_team = getattr(args, "wandb_team", wandb_team)
    args.wandb_group = getattr(args, "wandb_group", wandb_group)
    args.wandb_project = getattr(args, "wandb_project", wandb_project)

    wandb_project = wandb_project or args.wandb_project
    wandb_team = wandb_team or args.wandb_team
    wandb_group = wandb_group or args.wandb_group

    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=wandb_run_name,
        entity=wandb_team,
        config=args,
    )
    swanlab_init(args)


def _maybe_init_wandb_hydra(args):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("Wandb not configured, logging to console only")
        return

    args.wandb = True
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    wandb_team = os.getenv("WANDB_TEAM")
    wandb_group = os.getenv("WANDB_GROUP")

    target = getattr(args, "DistributedConfig", None) or getattr(
        args, "DistributedTrainConfig", None
    )
    if target is None:
        target = args

    target.wandb_team = getattr(args, "wandb_team", wandb_team)
    target.wandb_group = getattr(args, "wandb_group", wandb_group)
    target.wandb_project = getattr(args, "wandb_project", wandb_project)

    wandb_project = wandb_project or target.wandb_project
    wandb_team = wandb_team or target.wandb_team
    wandb_group = wandb_group or target.wandb_group

    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=wandb_run_name,
        wandb_team=wandb_team,
        config=args,
    )
