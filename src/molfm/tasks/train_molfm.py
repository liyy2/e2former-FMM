# -*- coding: utf-8 -*-
"""This script is used to fine-tune the molfm model on small molecules dataset, such MD17, MD22, PubChem50K, etc."""
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.optim.adamw import AdamW

from molfm.data.dataset import (
    MoleculeLMDBDataset,
    MultiDataset,
)

from molfm.logging import logger
from molfm.models.e2former.E2Former import E2FormerBackbone

from molfm.models.molfm_config import MOLFMConfig
from molfm.models.molfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from molfm.pipeline.schema import DistributedConfig, StepOutput
from molfm.pipeline.loop import TrainingLoop, CoreModule, seed_all
from molfm.tasks.heads import MOLFM_FT_REGISTER
from molfm.utils.runtime import cli, swanlab_init, wandb_init
from molfm.utils.runtime import set_env
from torch.optim import Adam

kcalmol_to_ev = 0.0433634

_MD22_SAMPLE_SIZE_DEFAULTS = {
    "at_at": 3000,
    "at_at_cg_cg": 2000,
    "stachyose": 8000,
    "dha": 8000,
    "ac_ala3_nhme": 6000,
    "buckyball_catcher": 600,
    "double_walled_nanotube": 800,
    "chig": 8000,
}

class Identity(nn.Module):
    def __init__(
        self,
        args: MOLFMConfig,
    ) -> None:
        super().__init__()
        self.args = args

    def forward(self, model_output, batched_data):
        return None, None


@dataclass
class SmallMolConfig(DistributedConfig, MOLFMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "e2former"
    train_val_test_split: List[float] = field(
        # default_factory=lambda: [0.2, 0.7, 0.1]
        default_factory=lambda: [0.9999, 0.00001, 0.0]
    )  # NOTE: This is only for MD data
    shuffle: bool = True
    # config
    data_path: str = ""
    encoder_ffn_embed_dim: int = 768
    loadcheck_path: str = ""
    use_pbc: bool = False
    energy_loss_weight: float = 0.1
    force_loss_weight: float = 0.9
    head_module: str = "md_energy_force_multi_head"  # if "", skip reset ft head
    loss_unit: str = "ev"
    loss_fn: str = "mae"
    data_path_list_valid: str = ""
    md22_protocol: bool = False
    md22_molecule: str = ""
    md22_sample_size: int = -1
    md22_train_prop: float = 0.95
    md22_seed: int = 48

cs = ConfigStore.instance()
cs.store(name="config_molfm_schema", node=SmallMolConfig)


def _resolve_md22_sample_size(
    md22_molecule: str,
    dataset_name: str,
    data_path: str,
    explicit_sample_size: int,
) -> int:
    if explicit_sample_size >= 0:
        return explicit_sample_size

    molecule_key = md22_molecule.lower().strip()
    dataset_key = dataset_name.lower().strip()
    path_key = data_path.lower()

    if molecule_key in _MD22_SAMPLE_SIZE_DEFAULTS:
        return _MD22_SAMPLE_SIZE_DEFAULTS[molecule_key]
    if dataset_key in _MD22_SAMPLE_SIZE_DEFAULTS:
        return _MD22_SAMPLE_SIZE_DEFAULTS[dataset_key]

    for key, value in _MD22_SAMPLE_SIZE_DEFAULTS.items():
        if key in path_key:
            return value

    return -1


def _split_md22_protocol(dataset, dataset_name: str, data_path: str, args):
    sample_size = _resolve_md22_sample_size(
        md22_molecule=args.md22_molecule,
        dataset_name=dataset_name,
        data_path=data_path,
        explicit_sample_size=args.md22_sample_size,
    )
    if sample_size < 0:
        raise ValueError(
            "MD22 protocol enabled but sample size is unknown. "
            "Set md22_sample_size explicitly or provide md22_molecule / dataset name "
            f"matching {_MD22_SAMPLE_SIZE_DEFAULTS.keys()}."
        )

    total_size = len(dataset)
    if sample_size > total_size:
        raise ValueError(
            f"MD22 sample_size ({sample_size}) is larger than dataset size ({total_size}) "
            f"for dataset '{dataset_name}'."
        )
    if not (0.0 < args.md22_train_prop < 1.0):
        raise ValueError(f"md22_train_prop must be in (0, 1), got {args.md22_train_prop}.")

    train_size = int(sample_size * args.md22_train_prop)
    valid_size = sample_size - train_size
    test_size = total_size - sample_size
    if train_size <= 0 or valid_size <= 0:
        raise ValueError(
            f"Invalid MD22 split sizes for sample_size={sample_size}, train_prop={args.md22_train_prop}: "
            f"train={train_size}, valid={valid_size}."
        )

    split_generator = torch.Generator().manual_seed(args.md22_seed)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=split_generator,
    )
    logger.info(
        "MD22 protocol split | dataset=%s | total=%d | sample_size=%d | train=%d | valid=%d | test=%d | seed=%d",
        dataset_name,
        total_size,
        sample_size,
        train_size,
        valid_size,
        test_size,
        args.md22_seed,
    )
    return train_dataset, valid_dataset, test_dataset


def load_data(args, extra_collate_fn=None):
    # Dataset will automatically load based on the args
    remove_atomref_energy = (
        args.remove_atomref_energy if hasattr(args, "remove_atomref_energy") else True
    )

    file_list = []
    file_list_valid = []
    dataset_name_list = args.dataset_name_list.split(",")
    dataset_split_raito = [float(i) for i in args.dataset_split_raito.split(",")]
    for sub_data_path in args.data_path_list.split(","):
        file_list.append(os.path.join(args.data_path, sub_data_path))
    
    if args.data_path_list_valid=="" or args.data_path_list_valid.lower()=="none":
        file_list_valid = ["" for _ in range(len(file_list))]
    else:
        for sub_data_path in args.data_path_list_valid.split(","):
            file_list_valid.append(
                os.path.join(args.data_path, sub_data_path)
            )
          
    assert (
            len(dataset_split_raito) == len(dataset_name_list) and 
            len(dataset_split_raito) == len(file_list_valid) and 
            len(dataset_split_raito) == len(file_list) 
        ), f"split ratio mismatch with number of datasets,{len(dataset_split_raito),len(dataset_name_list),len(file_list),len(file_list_valid)}"
      
    train_dataset_list = []
    valid_dataset_list = []
    test_dataset_list = []
    train_len = 0
    valid_len = 0
    test_len = 0
    
    for data_path, data_path_valid, dataset_name in zip(
        file_list, file_list_valid, dataset_name_list
    ):
        dataset = MoleculeLMDBDataset(
            args,
            data_path,
            data_name=dataset_name,
            remove_atomref_energy=remove_atomref_energy,
        )
        if data_path_valid:
            if args.md22_protocol:
                logger.warning(
                    "md22_protocol=True but explicit validation path is provided; "
                    "using explicit train/valid files and skipping MD22 protocol split."
                )
            train_dataset = dataset
            valid_dataset = MoleculeLMDBDataset(
                args,
                data_path_valid,
                data_name=dataset_name,
                remove_atomref_energy=remove_atomref_energy,
            )
        else:
            if args.md22_protocol:
                train_dataset, valid_dataset, test_dataset = _split_md22_protocol(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    data_path=data_path,
                    args=args,
                )
            else:
                shuffle = args.shuffle
                (
                    train_dataset,
                    valid_dataset,
                    test_dataset,
                ) = dataset.split_train_valid_test(
                    args.train_val_test_split, sort=False, shuffle=shuffle
                )
            test_dataset_list.append(test_dataset)
            test_len += len(test_dataset)
        train_dataset_list.append(train_dataset)
        valid_dataset_list.append(valid_dataset)
        train_len += len(train_dataset)
        valid_len += len(valid_dataset)
    print(
        f"dataset train_len : {train_len,train_dataset_list},dataset valid_len : {valid_len,valid_dataset_list},",
    )

    train_data = MultiDataset(
        args, train_dataset_list, train_len, extra_collate_fn=extra_collate_fn
    )
    valid_data = MultiDataset(
        args, valid_dataset_list, valid_len, extra_collate_fn=extra_collate_fn
    )
    test_data = (
        MultiDataset(
            args, test_dataset_list, test_len, extra_collate_fn=extra_collate_fn
        )
        if test_len > 0
        else None
    )

    return train_data, valid_data, test_data


class MOLFMModel_Atomic(CoreModule):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        not_init=False,
        molfm_finetune_head: nn.Module = None,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
    ):
        """
        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
            molfm_finetune_head: head used to finetune
        """

        super().__init__()
        if not_init:
            return

        self.molfm_config = MOLFMConfig(args)
        self.args = self.molfm_config.args
        if args.rank == 0:
            logger.info(self.args)

        if self.args.backbone == "e2former":
            self.decoder = E2FormerBackbone(**args.backbone_config)
            
        self.molfm_finetune_head = molfm_finetune_head
        self.checkpoint_loaded = self.reload_checkpoint()

        self.loss_fn = loss_fn(args)

    def reload_checkpoint(self):
        if self.molfm_config.molfm_finetune_mode:
            if os.path.exists(self.args.loadcheck_path):
                self.load_pretrained_weights(
                    self.args, checkpoint_path=self.args.loadcheck_path
                )
                loaded = True
                logger.info(f"checkpoint: {self.args.loadcheck_path} is loaded")
            else:
                logger.warning(
                    "Finetune or validation mode, but no checkpoint is loaded"
                )
                loaded = False
        else:
            logger.info("No checkpoint is loaded")
            loaded = False
        if self.molfm_config.molfm_finetune_mode:
            settings = dict(
                molfm_finetune_head=(
                    self.molfm_finetune_head.__class__ if self.molfm_finetune_head else None
                ),
            )
            logger.info(f"Finetune settings: {settings}")
        else:
            assert not self.molfm_finetune_head
            self.molfm_finetune_head = None

        return loaded

    def half(self):
        to_return = super().half()

        return to_return

    def _create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1)
        is_protein = (~is_periodic.unsqueeze(-1)) & (token_id > 129) & (token_id < 156)

        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein

        pos = batched_data["pos"]
        n_graphs, n_nodes = pos.size()[:2]

        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)
        batched_data["non_atom_mask"] = non_atom_mask

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        for key in list(checkpoints_state.keys()):
            if key.startswith("base."):
                checkpoints_state[key[5:]] = checkpoints_state.pop(key)
            if key.startswith("net."):
                checkpoints_state[key[4:]] = checkpoints_state.pop(key)

        # Warm-start Hybrid attention from a baseline checkpoint:
        # Baseline E2Former uses `blocks.*.ga.*` while Hybrid wraps the same local
        # attention as `blocks.*.ga.local_attn.*`. Remap those keys when possible.
        model_state = self.state_dict()
        if any(".ga.local_attn." in k for k in model_state.keys()):
            remapped = {}
            to_delete = []
            for k, v in list(checkpoints_state.items()):
                if (
                    ".ga." in k
                    and ".ga.local_attn." not in k
                    and ".ga.global_attn." not in k
                ):
                    k_local = k.replace(".ga.", ".ga.local_attn.", 1)
                    if (
                        k_local in model_state
                        and isinstance(v, torch.Tensor)
                        and model_state[k_local].shape == v.shape
                    ):
                        remapped[k_local] = v
                        to_delete.append(k)
            if remapped:
                for k in to_delete:
                    checkpoints_state.pop(k, None)
                checkpoints_state.update(remapped)
                logger.info(
                    "Remapped {} baseline attention keys to Hybrid local attention keys.",
                    len(remapped),
                )

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

        logger.info(f"checkpoint: {checkpoint_path} is loaded")

    def max_positions(self):
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

    def forward(self, batched_data, **unused):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """
        
        return self.forward_e2former(batched_data)

    def forward_e2former(
        self,
        batched_data,
        **unused,
    ):
        """
        Args:
            - batched_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """
        self._create_system_tags(batched_data)

        context = nullcontext()
        with context:
            pos = batched_data["pos"]
            batched_data["token_id"] == 0
            decoder_output = self.decoder(
                Data(**batched_data),
                # time_embed=time_embed,
            )

            decoder_x_output, decoder_vec_output = (
                decoder_output["node_featuresBxN"],
                decoder_output["node_vec_featuresBxN"],
            )
            n_graphs, n_nodes = pos.shape[:2]

            # atom mask to leave out unit cell corners for periodic systems
            non_atom_mask = torch.arange(
                n_nodes, dtype=torch.long, device=pos.device
            ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(
                -1
            )

            energy_per_atom = torch.zeros_like(batched_data["num_atoms"])
            # total_energy = torch.zeros_like(batched_data["num_atoms"])
            forces = torch.zeros_like(batched_data["pos"])
            noise_pred = torch.zeros_like(batched_data["pos"])
            autograd_forces = None

            result_dict = {
                "energy_per_atom": energy_per_atom,
                # "total_energy": total_energy,
                "forces": forces,
                # "aa_logits": None,  # aa_logits,
                # "time_step": None,
                # "noise_pred": noise_pred,
                "non_atom_mask": non_atom_mask,
                # "protein_mask": batched_data["protein_mask"],
                "is_molecule": batched_data["is_molecule"],
                "is_periodic": batched_data["is_periodic"],
                "is_protein": batched_data["is_protein"],
                # "is_complex": batched_data["is_complex"],
                # "is_seq_only": batched_data["is_seq_only"],
                "num_atoms": batched_data["num_atoms"],
                "pos": batched_data["pos"],
                "data_name": None
                if "data_name" not in batched_data
                else batched_data["data_name"],
            }

            if autograd_forces is not None:
                result_dict.update({"autograd_forces": autograd_forces})

            if self.molfm_config.molfm_finetune_mode:
                result_dict.update(
                    {
                        "decoder_x_output": decoder_x_output,
                        "decoder_vec_output": decoder_vec_output,
                    }
                )

                result_dict.update(decoder_output)

        if self.molfm_finetune_head:
            result_dict = self.molfm_finetune_head(result_dict)

        return result_dict

    def compute_loss(self, model_output, batched_data) -> StepOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            StepOutput: The model output which includes loss, log_output, num_examples.
        """
        bs = batched_data["pos"].size(0)
        loss, logging_output = self.loss_fn(model_output, batched_data)
        if self.molfm_finetune_head and hasattr(self.molfm_finetune_head, "update_loss"):
            loss, logging_output = self.molfm_finetune_head.update_loss(
                loss, logging_output, model_output, batched_data
            )
        return StepOutput(loss=loss, num_examples=bs, log_output=logging_output)

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)


@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_molfm"
)
def finetune(cfg: DictConfig) -> None:
    args = OmegaConf.to_object(cfg)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.save_dir,"pretrain_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    assert isinstance(args, SmallMolConfig)


    if getattr(args, "wandb", False):
        wandb_init(args)
    swanlab_init(args)
    seed_all(args.seed)
    set_env(args)

    head_module = None
    extra_collate_fn = None
    if len(args.head_module) > 0:
        if args.molfm_finetune_mode and args.head_module is not None:
            head_module = MOLFM_FT_REGISTER[args.head_module](args)
            extra_collate_fn = head_module.update_batched_data

    train_data, valid_data, test_data = load_data(
        args, extra_collate_fn=extra_collate_fn
    )

    # Define model
    model = MOLFMModel_Atomic(args, loss_fn=Identity, molfm_finetune_head=head_module)

    # Define optimizer
    optimizer = myAdam(
        model,
        impl=Adam if args.weight_decay == 0 else AdamW,
        lr=args.max_lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    lr_scheduler = groupWarmupDecayLR(
        optimizer,
        total_num_steps=args.total_num_steps,
        warmup_min_lr=args.min_lr,
        warmup_max_lr=args.max_lr,
        warmup_num_steps=args.warmup_num_steps,
        decay_type=DECAY_COSINE_RATE,
    )

    trainer = TrainingLoop(
        args,
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    
    if args.inference_mode:
        trainer.validate()
    else:
        trainer.train()
        trainer.resume()
        trainer.validate()


if __name__ == "__main__":
    try:
        finetune()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        pass
    try:
        import swanlab
        swanlab.finish()
    except Exception:
        pass
