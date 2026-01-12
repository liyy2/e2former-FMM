# -*- coding: utf-8 -*-
import multiprocessing
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
)

import molfm.models
from molfm.data.dataset import ProportionalSampler, SampleBatch
from molfm.logging import logger
from molfm.pipeline.schema import (
    RunState,
    StepOutput,
    ValidationLog,
)

def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b

class OutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return super().__getitem__(k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def items(self):
        return zip(self._fields, self)

    def keys(self):
        return self._fields

    def iget(self, idx: Union[int, slice]):
        """Select item(s) row-wise.

        Args:
            idx ([int, slice]): item to select

        Returns:
            Output of single item.
        """
        return self.__class__(*(x[idx] for x in self))


def transfer_to_device(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
    device: Union[str, torch.DeviceObjType],
    non_blocking: bool = False,
) -> Union[
    Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor],
]:
    """
    Move object to device.

    Args:
        x (dictionary of list of tensors): object (e.g. dictionary) of tensors to move to device
        device (Union[str, torch.DeviceObjType]): device, e.g. "cpu"

    Returns:
        x on targeted device
    """
    if isinstance(device, str):
        device = torch.device(device)
    return _move_any(x, device=device, non_blocking=non_blocking)


def _move_any(x, device, non_blocking=False):
    if isinstance(x, dict):
        return {name: _move_any(val, device=device, non_blocking=non_blocking) for name, val in x.items()}
    if is_dataclass(x):
        for f in fields(x):
            setattr(x, f.name, _move_any(getattr(x, f.name), device=device, non_blocking=non_blocking))
        return x
    if isinstance(x, OutputMixIn):
        for xi in x:
            _move_any(xi, device=device, non_blocking=non_blocking)
        return x
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking) if x.device != device else x
    if isinstance(x, (list, tuple)):
        return [_move_any(xi, device=device, non_blocking=non_blocking) for xi in x]
    return x


class CoreModule(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_loaded = False

    @abstractmethod
    def compute_loss(self, pred, batch) -> StepOutput:
        pass

    @abstractmethod
    def config_optimizer(
        self, model: Optional[torch.nn.Module]
    ) -> Tuple[Optimizer, LRScheduler]:
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            Tuple[Optimizer, LRScheduler]:
        """
        pass

    def before_training(self):
        """
        This method is called before training so you can do some initialization.
        For example, freeze some layers or set some layers to eval mode.
        """

        pass

    def after_training(self):
        """
        This method is called after training so you can do some finalization.
        """

        pass

    def before_batch(self):
        """
        This method is called before each batch so you can do some preprocessing.
        For example, set some layers to eval mode to disable dropout.
        """

        pass

    def after_batch(self):
        """
        This method is called after each batch so you can do some postprocessing.
        For example, set some layers to train mode to enable dropout.
        """

        pass

    def calculate_metric(self):
        """
        This method is called after each epoch to calculate the metric.
        """

        pass

    def reload_checkpoint(self) -> bool:
        """
        For compatibility with DDP, reload checkpoint in a model after DDP is called
        return True is a checkpoint is loaded (often used in finetuing case)
        """
        pass


def torch_compile(fn: molfm.models, state: bool) -> torch.compile:
    # check if cuda is available
    if not torch.cuda.is_available():
        logger.info("Torch.compile is disabled because cuda is not available.")
        return fn

    # individual set
    device_name = torch.cuda.get_device_name(0)
    fullgraph = False
    dynamic = None
    backend = "inductor"
    mode = "default"

    if state:
        logger.info(
            f"Torch.compile is enabled with "
            f"fullgraph:{fullgraph}, dynamic:{dynamic}, backend:{backend}, mode={mode}."
        )

        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.suppress_errors = True
        return torch.compile(
            fn.cuda() if torch.cuda.is_available() else fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            disable=(not state),
        )
    else:
        return fn


class FP16Scaler(object):
    def __init__(
        self,
        init_scale: int,
        scale_factor: float = 2.0,
        scale_interval: int = 1000,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.since_last_scale_up = 0
        self.scale_interval = scale_interval

    def check_grad_overflow(self, params) -> bool:
        for p in params:
            if p.grad is None:
                continue

            grad_norm = p.grad.data.norm()
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                return True

        return False

    def backward(self, loss):
        if self.enabled:
            scaled_loss = loss * self.scale
        else:
            scaled_loss = loss
        scaled_loss.backward()

    def unscale_and_clip_grad(self, model, clip_grad_norm: float, model_level=True):
        if clip_grad_norm > 0:
            if model_level:
                return torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.float()
                        p.grad.data /= self.scale

                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(p, clip_grad_norm)
        return None

    def step(self, model, optimizer, clip_grad_norm: float = 1.0):
        if self.enabled:
            if self.check_grad_overflow(model.parameters()):
                self.scale /= self.scale_factor
                logger.info(
                    f"Gradient overflow detected, reducing scale to {self.scale}"
                )
                self.since_last_scale_up = 0
                # Skip optimizer step
            else:
                gradient_norm = self.unscale_and_clip_grad(model, clip_grad_norm)
                optimizer.step()

                self.since_last_scale_up += 1
                if (
                    self.since_last_scale_up >= self.scale_interval
                    and self.scale < 2**15
                ):
                    self.scale *= self.scale_factor
                    self.since_last_scale_up = 0
        else:
            gradient_norm = self.unscale_and_clip_grad(model, clip_grad_norm)
            optimizer.step()
        return gradient_norm



class AccumulatingBatchIter(object):
    """
    This class is used to group batches into a larger batch. i.e., gradient accumulation.
    """

    def __init__(self, it, group_size, drop_last=False):
        self.it = it
        self.group_size = group_size
        self.drop_last = drop_last

    def __iter__(self):
        chunk = []
        for item in self.it:
            chunk.append(item)
            if len(chunk) == self.group_size:
                yield chunk
                chunk = []
        if not self.drop_last and chunk:
            yield chunk

    def __len__(self):
        if self.drop_last:
            return len(self.it) // self.group_size
        else:
            return (len(self.it) + self.group_size - 1) // self.group_size


class ExecutionEngine(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def train_step(self, grouped_batch_data: List[SampleBatch]) -> StepOutput:
        pass

    @abstractmethod
    def valid_step(self, batch_data: SampleBatch) -> ValidationLog:
        pass

    @abstractmethod
    def save_checkpoint(
        self, ckpt_id: Union[int, str], extra_state: Optional[dict] = None
    ):
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: RunState,
        model_states_only: bool = False,
    ) -> RunState:
        pass

    @abstractmethod
    def build_data_loader(self, train_data, val_data):
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def sync_valid_loss(self, total_loss, num_examples):
        pass

    @property
    @abstractmethod
    def grad_scale(self) -> float:
        pass

    def before_epoch(self, epoch: int):
        pass

    def skip_first_batches(self, start_iteration):
        return False

    def _accumulate_log_output(
        self, total_log_output, log_output, current_sample_count, num_new_samples
    ):
        for key in log_output:
            value = log_output[key]
            if key not in total_log_output:
                if isinstance(value, torch.Tensor):
                    total_log_output[key] = value.item()
                elif isinstance(value, tuple):
                    if isinstance(value[0], torch.Tensor):
                        v0 = value[0].item()
                    else:
                        v0 = value[0]
                    if isinstance(value[1], torch.Tensor):
                        v1 = value[1].item()
                    else:
                        v1 = value[1]
                    total_log_output[key] = (v0, v1)
                else:
                    total_log_output[key] = value
            else:
                if isinstance(value, torch.Tensor):
                    total_log_output[key] = safe_divide(
                        (
                            total_log_output[key] * current_sample_count
                            + value.item() * num_new_samples
                        ),
                        (current_sample_count + num_new_samples),
                    )
                elif isinstance(value, tuple):
                    if isinstance(value[0], torch.Tensor):
                        v0 = value[0].item()
                    else:
                        v0 = value[0]
                    if isinstance(value[1], torch.Tensor):
                        v1 = value[1].item()
                    else:
                        v1 = value[1]
                    v0_sum = total_log_output[key][0] * total_log_output[key][1]
                    v0_sum += v0 * v1
                    v1 += total_log_output[key][1]
                    v0 = safe_divide(v0_sum, v1)
                    total_log_output[key] = (v0, v1)
                else:
                    total_log_output[key] = safe_divide(
                        (
                            total_log_output[key] * current_sample_count
                            + value * num_new_samples
                        ),
                        (current_sample_count + num_new_samples),
                    )


class SingleProcessEngine(ExecutionEngine):
    def __init__(
        self, args, model: CoreModule, optimizer, lr_scheduler, device: str
    ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.world_size = 1

        if not torch.cuda.is_available():
            logger.warning("cuda is not available. use cpu instead")
            self.device = "cpu"
        self.scaler = FP16Scaler(
            init_scale=self.args.grad_scaler_init, enabled=self.args.fp16
        )

        if args.fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model = torch_compile(self.model, self.args.compile)

    @property
    def grad_scale(self) -> float:
        return self.scaler.scale

    def set_up(self):
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.model.config_optimizer()

    def barrier(self):
        pass

    def build_data_loader(
        self, train_data, valid_data
    ):
        train_batch_size_per_gpu = self.args.train_batch_size // (
            self.world_size * self.args.gradient_accumulation_steps
        )
        assert (
            train_batch_size_per_gpu > 0
        ), "train_batch_size_per_gpu should be greater than 0"

        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=train_batch_size_per_gpu,
            collate_fn=train_data.collate,
            drop_last=True,
            # pin_memory=True,
            # num_workers=32
        )

        if valid_data:
            valid_batch_size_per_gpu = self.args.val_batch_size // self.world_size
            assert (
                valid_batch_size_per_gpu > 0
            ), "valid_batch_size_per_gpu should be greater than 0"

            self.valid_data_loader = DataLoader(
                valid_data,
                sampler=None,
                batch_size=valid_batch_size_per_gpu,
                collate_fn=valid_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data: List[SampleBatch]) -> StepOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()

        self.optimizer.zero_grad()
        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        for batch_data in grouped_batch_data:
            self.model.before_batch()
            batch_data = transfer_to_device(batch_data, self.device)

            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
            loss = model_output.loss / len(grouped_batch_data)

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.warning("loss is nan or inf. skip this batch")
                loss = loss.new_tensor(0.0, requires_grad=True)
            else:
                success_batch_count += 1

            self.scaler.backward(loss)

            if model_output.num_examples is not None:
                self._accumulate_log_output(
                    total_log_output,
                    model_output.log_output,
                    sample_count,
                    model_output.num_examples,
                )
                sample_count += model_output.num_examples
                total_loss += model_output.loss * model_output.num_examples

            self.model.after_batch()

        if success_batch_count > 0:
            total_log_output["gradient_norm"] = (
                self.scaler.step(
                    self.model, self.optimizer, self.args.gradient_clipping
                ),
                1,
            )

        self.lr_scheduler.step()

        model_output.num_examples = sample_count
        model_output.loss = safe_divide(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def valid_step(self, batch_data: SampleBatch, epoch: int = 0) -> ValidationLog:
        self.model.eval()

        batch_data = transfer_to_device(batch_data, self.device)
        if self.args.AutoGradForce is True:
            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
        elif self.args.AutoGradForce is False:
            with torch.no_grad():
                pred = self.model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)

        if hasattr(batch_data, "batch_size"):
            num_examples = batch_data.batch_size
        elif hasattr(model_output, "num_examples"):
            num_examples = model_output.num_examples
        else:
            logger.info("num_examples is not found. set to None")
            num_examples = None

        return ValidationLog(
            valid_loss=model_output.loss.item(),
            epoch=epoch,
            num_examples=num_examples,
            logits=model_output.logits,
            label=model_output.label,
            extra_output=model_output.log_output,
        )

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_dir = Path(self.args.save_dir)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if self.args.fp16:
            checkpoint["fpscaler"] = self.scaler.scale

        if extra_state is not None:
            checkpoint.update(extra_state)
        logger.info("save checkpoint: {}", ckpt_id)
        torch.save(checkpoint, save_dir / ckpt_id)

        with open(save_dir / "checkpoint_list.txt", "a") as f:
            f.write("\n" + ckpt_id)

    def _transfer_optimizer_state_to_fp32(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if "exp_avg" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg"] = self.optimizer.state[p][
                        "exp_avg"
                    ].float()
                if "exp_avg_sq" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg_sq"] = self.optimizer.state[p][
                        "exp_avg_sq"
                    ].float()

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: RunState,
        model_states_only: bool = False,
    ) -> RunState:
        checkpoint_path = ckpt_dir / str(ckpt_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if not model_states_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if self.args.fp16:
                self.scaler.scale = checkpoint["fpscaler"]
                self._transfer_optimizer_state_to_fp32()

            logger.info(f"optimizer is loaded from checkpoint {ckpt_id}")

        if not model_states_only:
            for k, v in checkpoint.items():
                if k not in ["model", "optimizer", "lr_scheduler"]:
                    setattr(trainer_state, k, v)

        return trainer_state

    def sync_valid_loss(self, total_loss, num_examples):
        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            log_num_dict[k] = v.item()

        return {k: safe_divide(v, log_num_dict[k]) for k, v in log_dict.items()}

    def skip_first_batches(self, start_iteration):
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        else:
            return False


class DistributedEngine(SingleProcessEngine):
    def __init__(self, args, model: CoreModule, optimizer, lr_scheduler) -> None:
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda")

    def set_up(self):
        super().set_up()
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        ddp_timeout = os.environ.get("DDP_TIMEOUT_MINUTES", None)
        logger.critical(
            f"Initializing DDP by env://. word size: {self.world_size}, rank: {self.rank}, "
            f"local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}, "
            f"DDP_TIMEOUT_MINUTES: {ddp_timeout}"
        )
        torch.distributed.init_process_group(
            backend=self.args.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(minutes=int(ddp_timeout))
            if ddp_timeout is not None
            else timedelta(minutes=30),
        )

        torch.distributed.barrier()

        logger.success("DDP initialized.")

        self.model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
        )
        if self.model.checkpoint_loaded:
            logger.info("Reloading checkpoint after DDP to ensure correctness.")
            self.ddp_model.module.reload_checkpoint()

        self.ddp_model = torch_compile(self.ddp_model, self.args.compile)

    def barrier(self):
        torch.distributed.barrier()

    def train_step(self, grouped_batch_data: List[SampleBatch]) -> StepOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.ddp_model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        for idx, batch_data in enumerate(grouped_batch_data):
            self.model.before_batch()
            batch_data = transfer_to_device(batch_data, self.device)

            # No sync for gradient accumulation
            maybe_no_sync = (
                self.ddp_model.no_sync()
                if idx != len(grouped_batch_data) - 1
                else nullcontext()
            )

            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info("loss is nan or inf. skip this batch")
                    mask = torch.isnan(loss) | torch.isinf(loss)
                    loss[mask] = 0.0
                    self.scaler.backward(loss)
                else:
                    success_batch_count += 1
                    self.scaler.backward(loss)

            self._accumulate_log_output(
                total_log_output,
                model_output.log_output,
                sample_count,
                model_output.num_examples,
            )
            sample_count += model_output.num_examples
            total_loss += model_output.loss * model_output.num_examples
            self.model.after_batch()

        if success_batch_count > 0:
            total_log_output["gradient_norm"] = (
                self.scaler.step(
                    self.model, self.optimizer, self.args.gradient_clipping
                ),
                1,
            )
        self.lr_scheduler.step()
        model_output.num_examples = sample_count
        model_output.loss = safe_divide(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def build_data_loader(
        self, train_data, val_data
    ):
        train_batch_size_per_gpu = self.args.train_batch_size // (
            self.world_size * self.args.gradient_accumulation_steps
        )
        assert (
            train_batch_size_per_gpu > 0
        ), "train_batch_size_per_gpu should be greater than 0"

        if not isinstance(train_data, IterableDataset):
            if self.args.use_unified_batch_sampler:
                self.train_sampler = ProportionalSampler(
                    train_data,
                    self.args.dataset_split_raito,
                    self.args.dataset_micro_batch_size,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=self.args.seed,
                )
                self.train_data_loader = DataLoader(
                    train_data,
                    batch_sampler=self.train_sampler,
                    collate_fn=train_data.collate,
                    num_workers=self.args.unified_data_num_workers,
                    pin_memory=True,
                    persistent_workers=True
                    if self.args.unified_data_num_workers > 0
                    else False,
                    prefetch_factor=4
                    if self.args.unified_data_num_workers > 0
                    else None,
                )
            else:
                self.train_sampler = DistributedSampler(
                    train_data, num_replicas=self.world_size, rank=self.rank
                )
                self.train_data_loader = DataLoader(
                    train_data,
                    sampler=self.train_sampler,
                    batch_size=train_batch_size_per_gpu,
                    collate_fn=train_data.collate,
                    drop_last=True,
                )
        elif self.args.use_dali_pipeline:
            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=None,
                collate_fn=train_data.collate,
            )
        else:
            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=train_batch_size_per_gpu,
                collate_fn=train_data.collate,
                drop_last=True,
                num_workers=1,
            )

        if val_data:
            if self.args.use_unified_batch_sampler:
                valid_sampler = ProportionalSampler(
                    val_data,
                    self.args.dataset_split_raito,
                    self.args.dataset_micro_batch_size,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=self.args.seed,
                )
                self.valid_data_loader = DataLoader(
                    val_data,
                    batch_sampler=valid_sampler,
                    collate_fn=val_data.collate,
                )
            elif self.args.use_dali_pipeline:
                self.valid_data_loader = DataLoader(
                    val_data,
                    batch_size=None,
                    collate_fn=val_data.collate,
                )
            else:
                valid_batch_size_per_gpu = self.args.val_batch_size // self.world_size
                assert (
                    valid_batch_size_per_gpu > 0
                ), "valid_batch_size_per_gpu should be greater than 0"
                validsampler = torch.utils.data.distributed.DistributedSampler(
                    val_data, num_replicas=self.world_size, shuffle=False
                )
                self.valid_data_loader = DataLoader(
                    val_data,
                    sampler=validsampler,
                    batch_size=valid_batch_size_per_gpu,
                    collate_fn=val_data.collate,
                    drop_last=False,
                )
        else:
            self.valid_data_loader = None

    def before_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_examples = torch.Tensor([num_examples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

    def sync_valid_metric(self, label_list, logits_list):
        if not label_list or not logits_list:
            return None, None

        label = torch.cat(label_list, dim=0).to(self.device)
        logits = torch.cat(logits_list, dim=0).to(self.device)
        num_samples = torch.zeros(
            self.world_size + 1, device=self.device, dtype=torch.long
        )
        num_samples[self.rank + 1] = label.shape[0]
        torch.distributed.all_reduce(num_samples)
        total_samples = int(torch.sum(num_samples).item())
        for i in range(1, self.world_size + 1):
            num_samples[i] += num_samples[i - 1]
        total_label = torch.zeros(
            total_samples, *label.shape[1:], device=self.device, dtype=label.dtype
        )
        total_logits = torch.zeros(
            total_samples, *logits.shape[1:], device=self.device, dtype=logits.dtype
        )

        total_label[num_samples[self.rank] : num_samples[self.rank + 1]] = label
        total_logits[num_samples[self.rank] : num_samples[self.rank + 1]] = logits
        torch.distributed.all_reduce(total_label)
        torch.distributed.all_reduce(total_logits)
        return total_label, total_logits

    def calculate_metric(self, label, logits):
        return self.model.calculate_metric(label, logits)

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: safe_divide(v, log_num_dict[k]) for k, v in log_dict.items()}

    def skip_first_batches(self, start_iteration):
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        else:
            return False
