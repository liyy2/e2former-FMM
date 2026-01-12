# -*- coding: utf-8 -*-
import copy
import os
import random
import shutil
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Type, Union

import deepspeed
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from molfm.logging import logger, metric_logger
from molfm.pipeline.engine import (
    AccumulatingBatchIter,
    DistributedEngine,
    ExecutionEngine,
    SingleProcessEngine,
)
from molfm.pipeline.schema import (
    ParallelStrategy,
    RunConfig,
    RunState,
    TrainLog,
    ValidationLog,
)
from molfm.pipeline.engine import CoreModule

ENABLE_NNSCALER = False


def seed_all(seed):
    deepspeed.runtime.utils.set_random_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class RunningLoss:
    sum: float = 0.0
    num_examples: int = 0

    def add(self, loss, num_examples):
        loss_val, count = _normalize_loss_and_count(loss, num_examples)
        if count <= 0:
            return
        self.sum += loss_val * count
        self.num_examples += count

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples


@dataclass
class MetricStore:
    world_size: int = 1
    label_list: list = field(default_factory=list)
    logits_list: list = field(default_factory=list)

    def add(self, logits, label):
        if logits is None or label is None:
            return
        self.label_list.append(label)
        self.logits_list.append(logits)

    def reset(self):
        self.label_list = []
        self.logits_list = []


@dataclass
class LogStore:
    world_size: int = 1
    allreduce_fn: Optional[Callable] = None
    sum: float = 0.0
    num_examples: int = 0
    extra_log: dict = field(default_factory=dict)
    extra_log_num: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def __post_init__(self):
        self.extra_log.setdefault("total_acc_sample", 0)

    def add(self, loss, num_examples, extra_log=None):
        loss_val, count = _normalize_loss_and_count(loss, num_examples)
        if count <= 0:
            return
        self.sum += loss_val * count
        self.num_examples += count
        if extra_log:
            self._merge_extra(extra_log, count)

    def _merge_extra(self, extra_log, count):
        for k, v in extra_log.items():
            if k == "total_acc_sample" or not isinstance(v, (torch.Tensor, float, tuple)):
                continue
            weighted, weighted_num = _normalize_extra_value(v, count)
            if k not in self.extra_log:
                self.extra_log[k] = weighted
                self.extra_log_num[k] = weighted_num
            else:
                self.extra_log[k] += weighted
                self.extra_log_num[k] += weighted_num

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0
        self.start_time = time.time()
        for k in list(self.extra_log.keys()):
            if k == "total_acc_sample":
                continue
            self.extra_log[k] = 0.0
            self.extra_log_num[k] = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        if self.allreduce_fn is not None:
            log_dict = {"loss": self.sum}
            log_num_dict = {"loss": self.num_examples}
            reduced_loss_dict = self.allreduce_fn(log_dict, log_num_dict)
            return reduced_loss_dict["loss"]
        return self.sum / self.num_examples

    def _allreducelog(self, log_dict: dict = {}, log_num_dict: dict = {}):
        return self.allreduce_fn(log_dict, log_num_dict)

    @property
    def averge_log(self):
        self.extra_log["SamplePerSec"] = self.num_examples / (
            time.time() - self.start_time
        )
        self.extra_log_num["SamplePerSec"] = 1.0 / self.world_size

        self.extra_log["total_acc_sample"] /= self.world_size
        self.extra_log["total_acc_sample"] += self.num_examples
        self.extra_log_num["total_acc_sample"] = 1.0 / self.world_size

        if self.world_size == 1 or self.allreduce_fn is None:
            return {
                k: (v / self.extra_log_num[k]) if self.extra_log[k] else 0
                for k, v in self.extra_log.items()
            }
        reduced_log = self._allreducelog(self.extra_log, self.extra_log_num)
        self.extra_log["total_acc_sample"] = reduced_log["total_acc_sample"]
        return reduced_log


def _normalize_loss_and_count(loss, num_examples):
    if loss is None:
        return 0.0, 0
    if isinstance(loss, torch.Tensor):
        try:
            loss = loss.item()
        except Exception:
            logger.error(f"Loss is not a valid tensor: {loss}")
            loss = 0.0
    if isinstance(num_examples, torch.Tensor):
        num_examples = num_examples.item()
    if num_examples is None or num_examples <= 0:
        return 0.0, 0
    if np.isnan(loss) or np.isinf(loss):
        return 0.0, 0
    return float(loss), int(num_examples)


def _normalize_extra_value(value, count):
    if isinstance(value, torch.Tensor):
        return float(value.item()) * count, 1 * count
    if isinstance(value, tuple):
        return float(value[0]) * float(value[1]), float(value[1])
    return float(value) * count, 1 * count


class EarlyStopping:
    def __init__(self, patience=10, metric="valid_loss", mode="min"):
        self.patience = patience
        self.mode = mode
        if mode not in ["min", "max"]:
            raise ValueError("Mode for EarlyStopping should be min or max")
        self.metric = metric
        self.counter = 0
        self.should_stop = False
        if self.mode == "min":
            self.best = np.inf
        else:
            self.best = -np.inf
        self.best_at = 0
        self.history = []

    def __call__(self, value):
        if len(self.history) == 0:
            self.best = value
            self.best_at = 0
            self.history.append(value)
            return self.should_stop

        if (self.mode == "min" and value < self.best) or (
            self.mode == "max" and value > self.best
        ):
            self.best = value
            self.best_at = len(self.history)
            self.counter = 0
        else:
            self.counter += 1
        self.history.append(value)
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


class TrainingLoop(object):
    def __init__(
        self,
        args: RunConfig,
        model: Union[CoreModule, Type[CoreModule]],
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_log_dict: Optional[dict] = {},

        model_init: Callable[[], torch.nn.Module] = None,
        optimizer_init: Callable[
            [Iterator[torch.nn.Parameter]], Optimizer
        ] = None,
        lr_scheduler_init: Callable[
            [Optimizer], LRScheduler
        ] = None,
    ):
        super().__init__()
        self.args = args
        logger.info("Loop args: {}", args)

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            metric=args.early_stopping_metric,
            mode=args.early_stopping_mode,
        )
        self.model_init = model_init
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        if args.mm_tensorcore == "bf16":
            torch.set_float32_matmul_precision("medium")
        elif args.mm_tensorcore == "tf32":
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_float32_matmul_precision("highest")

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.accelerator = self.build_accelerator(loss_log_dict=loss_log_dict)
        self.accelerator.set_up()
        self.accelerator.build_data_loader(train_data, valid_data)

        self.state = RunState(args=args)

        self.save_dir = Path(self.args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint_dir = Path(
                self.args.finetune_from_checkpoint_dir
            )
        else:
            self.finetune_from_checkpoint_dir = None

        self.world_size = self.accelerator.world_size
        self.start_iteration = 0

        if args.profiling:
            assert torch.cuda.is_available(), "Profiling only works on GPU trainings"
            self.prof_dir = Path(args.prof_dir)
            self.prof_dir.mkdir(exist_ok=True)
            self.prof = self.profiler_init()

    def save_checkpoint(self, name: str, state: Union[RunState, dict]):
        if isinstance(state, RunState):
            self.accelerator.save_checkpoint(name, asdict(state))
        else:
            self.accelerator.save_checkpoint(name, state)
        self._save_rng_and_iter_state(self.save_dir)

    def _load_checkpoint(self, path: Path, model_states_only: bool = False):
        checkpoint_list_path = path / "checkpoint_list.txt"
        latest_path = path / "latest"  # latest path for DeepSpeed
        nnscaler_latest_path = None

        checkpoint_last = None
        if model_states_only and self.args.finetune_from_checkpoint_id is not None:
            checkpoint_last = self.args.finetune_from_checkpoint_id
        elif checkpoint_list_path.exists():
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]
        elif latest_path.exists():
            with open(latest_path, "r") as f:
                latest_list = f.read().splitlines()
            if len(latest_list) > 0:
                checkpoint_last = latest_list[-1]

        if checkpoint_last is not None:
            checkpoint_path = path / checkpoint_last
            if checkpoint_path.exists():
                if not model_states_only:
                    logger.info(f"Resume from checkpoint: {checkpoint_path}")
                else:
                    logger.info(f"Finetune from checkpoint: {checkpoint_path}")
                self.state = self.accelerator.load_checkpoint(
                    path,
                    checkpoint_last,
                    self.state,
                    model_states_only=model_states_only,
                )
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} does not exist.")
        else:
            logger.warning(
                f"Non-empty checkpoint_list.txt or latest file is not present in {path}, "
                f"or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
            )

    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint(self.save_dir)
        self.start_iteration = self._load_rng_and_iter_state(self.save_dir)
        logger.info(f"self.start_iteration = {self.start_iteration}.")

    def finetune_from_checkpoint(self):
        if self.finetune_from_checkpoint_dir is not None:
            self._load_checkpoint(
                self.finetune_from_checkpoint_dir, model_states_only=True
            )
        else:
            logger.warning("No finetune_from_checkpoint_dir is provided.")

    def build_accelerator(self, loss_log_dict: Optional[dict] = {}) -> ExecutionEngine:
        if self.args.strategy == ParallelStrategy.DDP:
            return DistributedEngine(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def build_log_output(self, loss, extra_output=None) -> TrainLog:
        try:
            lr = self.accelerator.lr_scheduler.get_last_lr()[0]
        except:
            lr = 0.0

        if type(loss) == torch.Tensor:
            loss = loss.item()

        return TrainLog(
            loss=loss,
            grad_scale=self.accelerator.grad_scale,
            lr=lr,
            epoch=self.state.epoch,
            batch=self.state.batch,
            total_samples=self.state.sample,
            global_step=self.state.global_step,
            extra_output=extra_output,
        )

    def should_stop(self) -> bool:
        if (
            self.args.total_num_epochs is not None
            and self.args.total_num_epochs > 0
            and self.state.epoch >= self.args.total_num_epochs
        ):
            return True
        if (
            self.args.total_num_steps is not None
            and self.args.total_num_steps > 0
            and self.state.global_step >= self.args.total_num_steps
        ):
            return True
        return False

    def should_save_batch_checkpoint(self) -> bool:
        return (
            self.args.save_batch_interval > 0
            and self.state.global_step % self.args.save_batch_interval == 0
        )

    def should_save_epoch_checkpoint(self) -> bool:
        return (
            self.args.save_epoch_interval > 0
            and self.state.epoch % self.args.save_epoch_interval == 0
        )

    def should_log(self) -> bool:
        return (
            self.args.log_interval > 0
            and self.state.global_step % self.args.log_interval == 0
        )

    def should_do_batch_validate(self) -> bool:
        return (
            self.args.val_batch_interval > 0
            and self.state.global_step % self.args.val_batch_interval == 0
        )

    def should_do_epoch_validate(self) -> bool:
        return (
            self.args.val_epoch_interval > 0
            and (self.state.epoch + 1) % self.args.val_epoch_interval == 0
        )

    @property
    def train_data_loader(self) -> DataLoader:
        """
        Return the training data loader.
        """
        return AccumulatingBatchIter(
            self.accelerator.train_data_loader,
            self.args.gradient_accumulation_steps,
            drop_last=True,
        )

    @property
    def valid_data_loader(self) -> DataLoader:
        return self.accelerator.valid_data_loader

    def count_parameters(self):
        # if self.args.strategy in [
        #     ParallelStrategy.Pipeline,
        #     ParallelStrategy.ThreeD,
        # ]:
        #     total_num = sum(p.numel() for p in self.accelerator.ppmodel.parameters())
        #     trainable_num = sum(
        #         p.numel()
        #         for p in self.accelerator.ppmodel.parameters()
        #         if p.requires_grad
        #     )
        # else:
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return total_num, trainable_num

    def train(self):
        """
        Train the model on the training data loader.
        """
        logger.info("Start training")
        logger.info(self.model)

        assert self.train_data_loader is not None

        if hasattr(self.model, "before_training"):
            self.model.before_training()
        if self.args.ifresume:
            self.resume()
        elif self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint()

        total_num, trainable_num = self.count_parameters()
        logger.info(
            "Total number of parameters: {:,} - number of trainable parameters: {:,}.",
            total_num,
            trainable_num,
        )

        with self.prof if self.args.profiling else nullcontext() as prof:
            while (
                self.state.epoch < self.args.total_num_epochs
                and self.state.global_step < self.args.total_num_steps
            ):
                self.accelerator.before_epoch(self.state.epoch)

                logger.info("Start Training for epoch: {}", self.state.epoch)

                loss_accumulator = RunningLoss()
                interval_loss_accumulator = LogStore(
                    self.accelerator.world_size, self.accelerator._allreducelog
                )

                # skip first batches
                data_iterator = iter(self.train_data_loader)
                data_iterator = self.skip_first_batches(
                    data_iterator, self.start_iteration
                )
                try:
                    for grouped_batch_data in data_iterator:
                        with torch.profiler.record_function("accelerator.train_step"):
                            model_output = self.accelerator.train_step(
                                grouped_batch_data
                            )
                        loss_accumulator.add(
                            model_output.loss, model_output.num_examples
                        )
                        interval_loss_accumulator.add(
                            model_output.loss,
                            model_output.num_examples,
                            model_output.log_output,
                        )

                        # Log and save checkpoint
                        self.state.batch += 1
                        self.state.global_step += 1
                        self.state.sample += model_output.num_examples

                        if self.should_do_batch_validate() and not self.args.profiling:
                            self.validate()

                        if self.should_log():
                            log_output = self.build_log_output(
                                interval_loss_accumulator.averge_loss,
                                interval_loss_accumulator.averge_log,
                            )
                            interval_loss_accumulator.reset()
                            metric_logger.log(
                                log_output, "train_inner", self.state.global_step
                            )

                        if self.should_save_batch_checkpoint():
                            checkpoint_name = (
                                f"checkpoint_E{self.state.epoch}_B{self.state.batch}.pt"
                            )
                            self.save_checkpoint(checkpoint_name, self.state)

                        if self.args.profiling:
                            prof.step()

                        if self.should_stop():
                            break
                except StopIteration:
                    logger.info("StopIteration")
                    pass

                log_output = self.build_log_output(loss_accumulator.averge_loss)
                metric_logger.log(log_output, "train", self.state.global_step)

                self.state.batch = 0

                self.accelerator.barrier()
                if self.should_save_epoch_checkpoint():
                    checkpoint_name = f"checkpoint_E{self.state.epoch}.pt"
                    self.save_checkpoint(checkpoint_name, self.state)

                if self.should_do_epoch_validate():
                    valid_log = self.validate()
                else:
                    valid_log = None
                self.accelerator.barrier()

                # if use early stopping
                if self.args.early_stopping:
                    if valid_log is None:
                        logger.warning(
                            "No validation log is available, early stopping is set but not not used in this epoch."
                        )
                    else:
                        value = getattr(valid_log, self.early_stopping.metric, None)
                        if value is None:
                            value = valid_log.extra_output.get(
                                self.early_stopping.metric, None
                            )
                        if value is None:
                            logger.warning(
                                f"Metric {self.early_stopping.metric} is not available in the "
                                f"validation log, early stopping is set but not not used in this epoch."
                            )
                        # update early stopping and save stop flag
                        should_stop = self.early_stopping(value)
                        # just updated or the first epoch, update the best checkpoint
                        if self.early_stopping.counter == 0:
                            if self.args.strategy in [
                                ParallelStrategy.DDP,
                            ]:
                                best_ckpt_path = (
                                    Path(self.args.save_dir)
                                    / f"checkpoint_E{self.early_stopping.best_at}.pt"
                                )
                            shutil.copy(
                                best_ckpt_path,
                                Path(self.args.save_dir) / "checkpoint_best.pt",
                            )
                        if should_stop:
                            logger.info(
                                f"Early stopping at epoch {self.state.epoch}, exiting training "
                                f"loop, copying best model to checkpoint_best.pt."
                            )
                            break

                        logger.info(
                            f"Best {self.early_stopping.metric} is {self.early_stopping.best} "
                            f"at epoch {self.early_stopping.best_at}, "
                            f"not improving over past {self.early_stopping.counter} epochs. "
                        )

                if self.should_stop():
                    break
                self.state.epoch += 1
            if hasattr(self.model, "after_training"):
                self.model.after_training()

        # profiling results
        if self.args.profiling:
            self.profiler_end(self.prof)

        logger.info("Finished Training")

    def validate(self):
        """
        Validate the model on the validation data loader.
        """
        if self.valid_data_loader is None:
            logger.warning("No validation data, skip validation")
            return

        logger.info(
            "Start validation for epoch: {}, global step: {}",
            self.state.epoch,
            self.state.global_step,
        )

        loss_accumulator = RunningLoss()
        interval_loss_accumulator = LogStore(
            self.accelerator.world_size, self.accelerator._allreducelog
        )

        if self.args.calculate_metrics:
            metric_accumulator = MetricStore(
                self.accelerator.world_size,
            )

        for idx, batch_data in enumerate(self.valid_data_loader):
            if self.args.AutoGradForce is True:
                output = self.accelerator.valid_step(batch_data, epoch=self.state.epoch)
            elif self.args.AutoGradForce is False:
                with torch.no_grad():
                    output = self.accelerator.valid_step(
                        batch_data, epoch=self.state.epoch
                    )
            loss_accumulator.add(output.valid_loss, output.num_examples)
            interval_loss_accumulator.add(
                output.valid_loss,
                output.num_examples,
                output.extra_output,
            )

            if self.args.calculate_metrics:
                metric_accumulator.add(
                    output.logits,
                    output.label,
                )

            if (idx + 1) % self.args.val_batch_log_interval == 0:
                logger.info(
                    "Validtion batch: {} / {}, loss: {}",
                    idx + 1,
                    len(self.valid_data_loader),
                    output.valid_loss,
                )
                if self.args.val_batch_log_all_metric:
                    interval_loss_accumulator_for_log = deepcopy(
                        interval_loss_accumulator
                    )
                    valid_log = ValidationLog(
                        valid_loss=output.valid_loss,
                        num_examples=output.num_examples,
                        epoch=self.state.epoch,
                        extra_output={
                            **interval_loss_accumulator_for_log.averge_log,
                            **dict(),
                        },
                    )
                    metric_logger.log(
                        valid_log, "valid", self.state.global_step, log_wandb=False
                    )

        # DDP and Zero need to sync loss and num_examples at validation
        total_loss, num_examples = self.accelerator.sync_valid_loss(
            loss_accumulator.sum, loss_accumulator.num_examples
        )

        if self.args.calculate_metrics:
            label, logits = self.accelerator.sync_valid_metric(
                metric_accumulator.label_list, metric_accumulator.logits_list
            )
            metric_accumulator.reset()
            metric_results = self.accelerator.calculate_metric(label, logits)
        else:
            metric_results = dict()

        if num_examples > 0:
            valid_loss = total_loss / num_examples
        else:
            valid_loss = 0

        valid_log = ValidationLog(
            valid_loss=valid_loss,
            num_examples=num_examples,
            epoch=self.state.epoch,
            extra_output={**interval_loss_accumulator.averge_log, **metric_results},
        )
        metric_logger.log(valid_log, "valid", self.state.global_step)
        return valid_log

    def _save_rng_and_iter_state(self, checkpoint):
        """
        Save the RNG and iteration states to the checkpoint to resume training from break point.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.random.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "iteration": self.state.batch,
            "epoch": self.state.epoch,
        }

        if self.accelerator.world_size > 1:
            process_index = self.args.rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")

        torch.save(rng_state, rng_file)

    def _load_rng_and_iter_state(self, checkpoint):
        """
        Load the RNG and iteration states from the checkpoint to resume training from break point.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        if self.accelerator.world_size > 1:
            process_index = self.args.rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.warning(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.warning(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        try:
            random.setstate(checkpoint_rng_state["python"])
        except:
            logger.warning(
                "Didn't manage to set back the RNG states of the Python random module, this won't yield the same "
                "results as if the training had not been interrupted."
            )
        try:
            np.random.set_state(checkpoint_rng_state["numpy"])
        except:
            logger.warning(
                "Didn't manage to set back the RNG states of the Numpy random module, this won't yield the same "
                "results as if the training had not been interrupted."
            )
        try:
            torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        except:
            logger.warning(
                "Didn't manage to set back the RNG states of the CPU, this won't yield the same results as if the "
                "training had not been interrupted."
            )
        if torch.cuda.is_available():
            if self.accelerator.world_size > 1:
                try:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.warning(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.warning(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

        if "epoch" in checkpoint_rng_state:
            self.state.epoch = checkpoint_rng_state["epoch"]

        start_iteration = checkpoint_rng_state["iteration"]

        return start_iteration

    def skip_first_batches(self, data_iterator, start_iteration=None):
        """
        Skip the first start_iteration batches in the training data loader to resume training from break point.
        Args:
            start_iteration (int): the number of batches to skip
        """

        if start_iteration is None or start_iteration == 0:
            return data_iterator

        self.state.batch = start_iteration

        if self.args.use_unified_batch_sampler:
            skip_first_batches_in_accelerator = self.accelerator.skip_first_batches(
                start_iteration
            )
            if skip_first_batches_in_accelerator:
                self.start_iteration = 0
                return iter(self.train_data_loader)
        else:
            skip_first_batches_in_accelerator = False

        if not skip_first_batches_in_accelerator:
            logger.info(f"Skipping the first {start_iteration} batches")
            for i, _ in tqdm(
                enumerate(data_iterator),
                desc=f"Skipping first {start_iteration} batches",
                miniters=1000,
            ):
                if i == start_iteration - 1:
                    break

            self.start_iteration = 0
            return data_iterator

    def profiler_init(self) -> torch.profiler.profile:
        # Torch profiler ReadMe: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=1, skip_first=0
            ),
            # custom profiling results (TB in Torch ReadMe:)
            # https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md
            on_trace_ready=(
                torch.profiler.tensorboard_trace_handler(
                    str(self.prof_dir / "tensorboard")
                )
                if self.args.ptensorboard
                else self.custom_trace_handler
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )

    def custom_trace_handler(self, prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(self.prof_dir / "profiler_chrome.json"))
        """
        stack export is not working as of 21.02.24:
        https://github.com/pytorch/pytorch/issues/100253
        """
        prof.export_stacks(
            str(self.prof_dir / "profiler_stacks.txt"), metric="self_cuda_time_total"
        )
        logger.info("Profiler called a trace.")

    def profiler_end(self, prof: torch.profiler.profile) -> None:
        if self.args.rank == 0:
            with (self.prof_dir / "profiler_list.dat").open(
                "w", encoding="utf-8"
            ) as outT:
                outT.write(
                    f"Profiler results:\n{prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1)}"
                )
                outT.write(
                    "\n--------------------------------------------------------\n"
                )
                outT.write(
                    f"Memory requirement: {str(int(torch.cuda.max_memory_reserved(self.args.rank)/1024/1024))} MB"
                )
                outT.write(
                    "\n--------------------------------------------------------\n"
                )
                outT.write(
                    f"Memory summary:\n{str(torch.cuda.memory_summary(self.args.rank))}"
                )
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )
        logger.info(
            f"Memory summary:\n{str(torch.cuda.memory_summary(self.args.rank))}"
        )

        if self.args.strategy.find("Zero") == 0:
            deepspeed.comm.log_summary()
