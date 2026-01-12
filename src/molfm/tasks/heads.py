# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Union

import e3nn
import torch
import torch.nn as nn
from torch import Tensor
from molfm.models.molfm_config import MOLFMConfig
from molfm.logging.loggers import logger

kcalmol_to_ev = 0.0433634

class GradientHead(torch.nn.Module):
    def __init__(
        self,
        molfm_config: MOLFMConfig,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_energy_std=1.0,
        periodic_energy_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
        periodic_stress_mean=0.0,
        periodic_stress_std=1.0,
        supervise_total_energy: bool = False,
    ):
        super(GradientHead, self).__init__()
        self.molfm_config = molfm_config
        self.molecule_energy_per_atom_std = molecule_energy_per_atom_std
        self.periodic_energy_per_atom_std = periodic_energy_per_atom_std
        self.molecule_energy_std = molecule_energy_std
        self.periodic_energy_std = periodic_energy_std
        self.molecule_force_std = molecule_force_std
        self.periodic_force_std = periodic_force_std
        self.periodic_stress_mean = periodic_stress_mean
        self.periodic_stress_std = periodic_stress_std
        self.supervise_total_energy = supervise_total_energy
        self.strain = None

    def wrap_input(self, batched_data):
        pos: Tensor = batched_data["pos"]
        pos.requires_grad_(True)
        batch_size = pos.size(0)
        device = pos.device
        dtype = pos.dtype

    def forward(
        self,
        energy_per_atom,
        non_atom_mask,
        pos,
        cell,
        is_periodic,
        is_molecule,
    ):
        energy_per_atom = energy_per_atom.masked_fill(non_atom_mask, 0.0)
        energy_per_atom = torch.where(
            is_periodic.unsqueeze(-1),
            energy_per_atom
            * (
                self.periodic_energy_std
                if self.supervise_total_energy
                else self.periodic_energy_per_atom_std
            ),
            energy_per_atom,
        )
        energy_per_atom = torch.where(
            is_molecule.unsqueeze(-1),
            energy_per_atom
            * (
                self.molecule_force_std
                if self.supervise_total_energy
                else self.molecule_energy_per_atom_std
            ),
            energy_per_atom,
        )
        energy = energy_per_atom.sum(dim=-1, keepdim=True)
        grad_outputs = [torch.ones_like(energy)]

        grad = torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=grad_outputs,
            create_graph=self.training,
            retain_graph=True,
        )

        force_grad = grad[0]
        forces = torch.neg(force_grad)

        forces = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1),
            forces / self.periodic_force_std,
            forces,
        )
        forces = torch.where(
            is_molecule.unsqueeze(-1).unsqueeze(-1),
            forces / self.molecule_force_std,
            forces,
        )

        stress = None

        self.strain = None

        return forces, stress

class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict:
            raise ValueError("Key %s is not in registry %s." % (key, self._name))
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

class MolFMBaseModule(nn.Module):
    def __init__(self, args: MOLFMConfig):
        super().__init__()
        self.args = args

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        return loss, logging_output

MOLFM_FT_REGISTER: Union[Dict[str, MolFMBaseModule.__class__], Register] = Register(
    "molfm_finetine_module_register"
)

class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class_mapping = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(-2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v

class EquivariantVectorOutput(nn.Module):
    def __init__(self, hidden_channels=768, activation="silu"):
        super(EquivariantVectorOutput, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze(-1)

@MOLFM_FT_REGISTER.register("md_energy_force_multi_head")
class MDEnergyForceMultiHead(MolFMBaseModule):
    def __init__(self, args: MOLFMConfig):
        super().__init__(args)

        self.loss_fn = args.loss_fn.lower()
        self.energy_loss_weight = args.energy_loss_weight
        self.force_loss_weight = args.force_loss_weight
        self.auto_grad = args.AutoGradForce

        if args.backbone == "e2former":
            embedding_dim = e3nn.o3.Irreps(
                args.backbone_config["irreps_node_embedding"]
            )[0][0]
        else:
            embedding_dim = args.encoder_embed_dim

        # Initialize multiple energy and force heads based on dataset_name_list
        self.energy_heads = nn.ModuleDict()
        self.force_heads = nn.ModuleDict()

        self.dataset_rename = {
            "deshaw_120":"deshaw",
            "deshaw_400":"deshaw",
            "GEMS_water":"GEMS",
            "pm6-wb97xd3": "fchem",
            "geom": "fchem",
            "fchem_pbc": "fchem",
            "fchem_surf": "fchem",
            "fchem_rare": "fchem",
            "fchem_pept": "fchem",
            "fchem_pept_drug":"fchem",
            "fchem_pept_drug_sph":"fchem",
            "fchem_pept_pept_sph":"fchem",
            "fchem_water":"fchem",
        }

        for dataset_name in args.dataset_name_list.split(","):
            # Create energy head for each dataset_name (deshaw_120: engergy head + autograd)
            head_name = (
                dataset_name
                if dataset_name not in self.dataset_rename
                else self.dataset_rename[dataset_name]
            )
            if head_name not in self.energy_heads:
                self.energy_heads[head_name] = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim, bias=True),
                    nn.SiLU(),
                    nn.Linear(embedding_dim, 1, bias=True),
                )

                if not self.auto_grad:
                    self.force_heads[head_name] = EquivariantVectorOutput(embedding_dim)
                else:
                    self.force_heads[head_name] = GradientHead(args)

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "pred_energy" in result_dict:return result_dict
        
        decoder_x_output = result_dict["decoder_x_output"]
        decoder_vec_output = result_dict["decoder_vec_output"]
        dataset_name = result_dict["data_name"][0]
        # Use the correct energy head based on dataset_name
        head_name = (
            dataset_name
            if dataset_name not in self.dataset_rename
            else self.dataset_rename[dataset_name]
        )
        energy_out = self.energy_heads[head_name](decoder_x_output).squeeze(-1)
        energy_out = energy_out.masked_fill(result_dict["non_atom_mask"], 0.0)
        result_dict["pred_energy"] = energy_out.sum(dim=-1)
        result_dict["pred_energy_per_atom"] = (
            result_dict["pred_energy"] / result_dict["num_atoms"]
        )

        # If dataset_name is not "deshaw_400" or "deshaw_650", process force head
        # if dataset_name not in ["deshaw_400", "deshaw_650"]:
        if dataset_name not in ["deshaw_no_force"]:
            if not self.auto_grad or (head_name in ["deshaw_650", "deshaw_bba"]):
                force_out = self.force_heads[head_name](
                    decoder_x_output, decoder_vec_output
                ).squeeze(-1)
            else:
                force_out, _ = self.force_heads[head_name](
                    energy_out,
                    result_dict["non_atom_mask"],
                    result_dict["pos"],
                    None,  # cell only used for stress autograd version
                    result_dict["is_periodic"],
                    result_dict["is_molecule"],
                )

            expanded_mask = (
                result_dict["non_atom_mask"].unsqueeze(-1).expand_as(force_out)
            )
            result_dict["pred_forces"] = force_out.masked_fill(expanded_mask, 0.0)

        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        # Extract information from batched data
        data_name = batched_data["data_name"][0]
        model_output["pred_energy"]
        f_pred = model_output.get("pred_forces", None)
        e_pred_peratom = model_output["pred_energy_per_atom"]

        e_true = batched_data["energy"]
        f_true = batched_data.get("forces", None)
        e_true_peratom = batched_data["energy_per_atom"]

        # Unit conversion if needed
        if self.args.loss_unit == "kcal/mol":
            # e_true /= kcalmol_to_ev
            e_true_peratom /= kcalmol_to_ev
            if f_true is not None:
                f_true /= kcalmol_to_ev
        elif (self.args.loss_unit).lower() == "mev":
            # e_true /= 0.001
            e_true_peratom /= 0.001
            if f_true is not None:
                f_true /= 0.001
        # Calculate energy loss
        if self.loss_fn == "mae":
            # e_loss = torch.mean(torch.abs(e_pred - e_true))
            e_peratom_loss = torch.mean(torch.abs(e_pred_peratom - e_true_peratom))

        elif self.loss_fn == "mse":
            # e_loss = torch.mean((e_pred - e_true) ** 2)
            e_peratom_loss = torch.mean((e_pred_peratom - e_true_peratom) ** 2)
        loss = self.energy_loss_weight * e_peratom_loss
        logging_output = {}
        # Batch size
        size = e_true.shape[0]
        if f_pred is not None and f_true is not None:
            if f_pred.shape != f_true.shape:
                f_true = f_true[~model_output["non_atom_mask"]]
                mask_weight = 1.0
            else:
                mask_weight = model_output["non_atom_mask"].numel() / (
                    model_output["non_atom_mask"].numel()
                    - torch.sum(model_output["non_atom_mask"].float())
                    )
            f_mae = torch.mean(torch.abs(f_pred.detach() - f_true))

            if self.loss_fn == "mae":
                f_loss = torch.mean(torch.abs(f_pred - f_true))
            elif self.loss_fn == "mse":
                f_loss = torch.mean((f_pred - f_true) ** 2)
            loss += self.force_loss_weight * f_loss

            # Calculate force magnitudes
            # Compute magnitudes of each force vector and find the maximum for each sample
            f_loss_magnitudes = torch.norm(
                torch.abs(f_pred - f_true), dim=-1
            )  # Magnitude of each (x, y, z) force vector
            max_f_loss = torch.max(
                f_loss_magnitudes
            )  # Overall maximum force discrepancy


            # cos{(0,0,0),(0,0,0)} = 0, thus div mask_weight
            forces_cosine_sim = torch.mean(
                torch.cosine_similarity(f_pred.detach(), f_true, dim=-1, eps=1e-8)
            )
            logging_output["force_loss"] = (f_mae * mask_weight, size)
            logging_output[f"force_loss_{data_name}"] = (f_mae * mask_weight, size)
            logging_output["max_f_loss"] = (
                max_f_loss,
                size,
            )  # Log the mean of the maximum forces
            logging_output[f"max_f_loss{data_name}"] = (
                max_f_loss,
                size,
            )  # Log the mean of the maximum forces
            logging_output[f"f_cos_{data_name}"] = (
                forces_cosine_sim * mask_weight,
                size,
            )  # Log the mean of the maximum forces



        e_mae = torch.mean(
            batched_data["num_atoms"] * torch.abs(e_pred_peratom - e_true_peratom)
        )
        # e_mae = torch.mean(torch.abs(e_pred.detach() - e_true))
        e_peratom_mae = torch.mean(torch.abs(e_pred_peratom.detach() - e_true_peratom))


        # Update logging output
        logging_output.update(
            {
                "loss": loss,
                "energy_loss": (e_mae, size),
                "energy_peratom_loss": (e_peratom_mae, size),
                f"energy_loss_{data_name}": (e_mae, size),
                f"energy_peratom_loss_{data_name}": (e_peratom_mae, size),
            }
        )

        return loss, logging_output
