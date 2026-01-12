# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import torch


class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"
    RELATIVE_POS_FLIP_INVAR: str = "RELATIVE_POS_FLIP_INVAR"
    AUGMENTED_RELATIVE_POS: str = "AUGMENTED_RELATIVE_POS"
    RELATIVE_POS_VEC_BIAS: str = "RELATIVE_POS_VEC_BIAS"

    def __str__(self):
        return self.value


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class ForceLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class StressLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


class ForceHeadType(Enum):
    LINEAR: str = "LINEAR"
    GATED_EQUIVARIANT: str = "GATED_EQUIVARIANT"
    MLP: str = "MLP"

    def __str__(self) -> str:
        return self.value


class GaussianFeatureNodeType(Enum):
    EXCHANGABLE: str = "EXCHANGABLE"
    NON_EXCHANGABLE: str = "NON_EXCHANGABLE"
    NON_EXCHANGABLE_DIFF_SELF_EDGE: str = "NON_EXCHANGABLE_DIFF_SELF_EDGE"

    def __str__(self) -> str:
        return self.value

@dataclass
class MOLFMConfig:
    model_type: str = "molfm"
    decoder_ffn_dim: int = 2048
    decoder_hidden_dim: int = 512

    task: str = "mae"
    sample_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""

    data_path_list: str = ""
    dataset_name_list: str = ""
    dataset_split_raito: str = ""
    dataset_micro_batch_size: str = ""

    share_attention_bias: bool = False

    AutoGradForce: bool = False

    molfm_finetune_mode: bool = False
    inference_mode: bool = False
    molfm_finetune_skip_ori_head: bool = False
    # charge_embedding
    use_charge_embedding: bool = False

    encoder_embed_dim: int = 256
    embedding_dim: int = 256
    ffn_embedding_dim: int = 768


    def __init__(
        self,
        args,
        **kwargs,
    ):
        self.args = args
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
