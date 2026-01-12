# -*- coding: utf-8 -*-
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import bisect
import glob
import math
import pickle as pkl
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torch_cluster import radius_graph

from molfm.data.collator import pack_batch
from molfm.data.utils import dataset_profile
from molfm.logging import logger
from molfm.models.molfm_config import MOLFMConfig

@dataclass
class SampleRecord:
    pass

@dataclass
class SampleBatch(SampleRecord):
    batch_size: int

class MoleculeLMDBDataset(ABC):

    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """

    energy = "energy"
    forces = "forces"
    _CELL_CORNER_MATRIX = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        args: MOLFMConfig,
        path,
        data_name="deshaw_di_mol",
        remove_atomref_energy=True,
    ):
        super(MoleculeLMDBDataset, self).__init__()

        self.data_name = data_name
        (
            self.atom_reference,
            self.system_ref,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            self.has_energy,
            self.has_forces,
            self.pbc,
            self.unit,
            others,
        ) = dataset_profile(data_name)
        self.is_pbc = sum(self.pbc) > 0
        self.pbc = torch.Tensor(self.pbc).bool()
        db_paths = self._collect_lmdb_paths(path)
        assert len(db_paths) > 0, "No LMDBs found"
        self._keys, self.envs = [], []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.remove_atomref_energy = remove_atomref_energy
        self.args = args

    def _collect_lmdb_paths(self, path):
        db_paths = []
        if isinstance(path, str):
            db_paths.extend(self._expand_path(path))
        elif isinstance(path, list):
            for p in path:
                db_paths.extend(self._expand_path(p))
        return db_paths

    @staticmethod
    def _expand_path(path):
        if path.endswith("lmdb"):
            return [path]
        return glob.glob(path + "/*.lmdb")

    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = self.envs[-1].begin().get("length".encode("ascii"))
            if length is not None:
                length = pkl.loads(length)
            else:
                length = self.envs[-1].stat()["entries"]

            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        db_idx, el_idx = self._resolve_db_index(idx)
        data_object = self._load_data_object(db_idx, el_idx)
        data_object.id = el_idx

        energy = self._extract_energy(data_object)
        if self.remove_atomref_energy:
            energy = self._remove_atom_reference(energy, data_object.atomic_numbers)
        energy = torch.Tensor([energy]).reshape(-1)

        out = self._build_base_output(idx, data_object, energy)
        if self.is_pbc:
            out = self._apply_pbc(data_object, out)
        else:
            out = self._apply_non_pbc(out, energy)
        out = self._cast_float_fields(out)
        out = self.build_2d_graph(out)
        return out

    def _resolve_db_index(self, idx):
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0
        return db_idx, el_idx

    def _load_data_object(self, db_idx, el_idx):
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        from fairchem.core.common.utils import pyg2_data_transform

        return pyg2_data_transform(pkl.loads(datapoint_pickled))

    @staticmethod
    def _extract_energy(data_object):
        if "energy" not in data_object:
            return data_object.y
        return data_object.energy

    def _remove_atom_reference(self, energy, atomic_numbers):
        unique, counts = np.unique(atomic_numbers.int().numpy(), return_counts=True)
        energy = energy - np.sum(self.atom_reference[unique] * counts)
        return torch.Tensor([energy - self.system_ref])

    def _build_base_output(self, idx, data_object, energy):
        mapped_charge = self._map_charge(data_object)
        cluster_ids = self._map_cluster_ids(data_object)
        return {
            "sample_type": 0,
            "coords": data_object.pos,
            "forces": data_object.force * self.unit
            if "force" in data_object
            else data_object.forces * self.unit,
            "num_atoms": data_object.pos.shape[0],
            "token_type": data_object.atomic_numbers.int().reshape(-1),
            "idx": idx,
            "energy": energy.reshape(-1) * self.unit,
            "has_energy": torch.tensor([self.has_energy], dtype=torch.bool),
            "has_forces": torch.tensor([self.has_forces], dtype=torch.bool),
            "data_name": self.data_name,
            "charge": mapped_charge.reshape(-1),
            "cluster_ids": cluster_ids.reshape(-1),
            "cluster_centers": data_object.cluster_centers
            if "cluster_centers" in data_object
            else torch.tensor([0], dtype=torch.int).unsqueeze(-1),
        }

    @staticmethod
    def _map_charge(data_object):
        if "charge" in data_object.keys():
            return (data_object.charge + 5).int()
        return torch.tensor([0], dtype=torch.int)

    @staticmethod
    def _map_cluster_ids(data_object):
        if "cluster_ids" in data_object.keys():
            return data_object.cluster_ids.int()
        return torch.tensor([0], dtype=torch.int)

    def _apply_pbc(self, data_object, out):
        out["token_type"] = torch.cat(
            [out["token_type"], torch.full([8], 128)], dim=-1
        )
        cell_corner_pos = torch.matmul(
            self._CELL_CORNER_MATRIX, data_object.cell.squeeze(dim=0).float()
        )
        out["coords"] = torch.cat([out["coords"], cell_corner_pos], dim=0)
        out["forces"] = torch.cat(
            [
                torch.tensor(out["forces"].clone().detach(), dtype=torch.float32),
                torch.zeros([8, 3], dtype=torch.float32),
            ],
            dim=0,
        )
        out["cell"] = data_object.cell.squeeze(dim=0)
        out["pbc"] = self.pbc
        out["stress"] = torch.zeros((3, 3), dtype=torch.float32, device=out["energy"].device)
        if hasattr(data_object, "cell_offsets"):
            out["cell_offsets"] = torch.tensor(data_object.cell_offsets.numpy())
        out["energy_per_atom"] = out["energy"] / out["num_atoms"]
        return out

    def _apply_non_pbc(self, out, energy):
        out["cell"] = torch.zeros((3, 3), dtype=torch.float32)
        out["pbc"] = torch.zeros(3, dtype=torch.float32).bool()
        out["stress"] = torch.zeros((3, 3), dtype=torch.float32, device=energy.device)
        out["energy_per_atom"] = out["energy"] / out["num_atoms"]
        return out

    @staticmethod
    def _cast_float_fields(out):
        float_exclude = {
            "num_atoms",
            "token_type",
            "idx",
            "edge_index",
            "has_energy",
            "has_forces",
            "sample_type",
            "data_name",
        }
        for key in list(out.keys()):
            if key not in float_exclude:
                out[key] = out[key].clone().detach().float()
        return out

    def build_empty_graph(self, data):
        N = data["num_atoms"]

        # Initialize adj as a zero matrix of the correct shape
        adj = torch.zeros([N, N], dtype=torch.bool)

        # Initialize edge_index and edge_attr as zero matrices
        data["edge_index"] = torch.zeros(
            (2, 0), dtype=torch.long
        )  # Empty edge index with correct shape
        data["edge_attr"] = torch.zeros(
            (0, 1), dtype=torch.long
        )  # Empty edge attribute with correct shape

        # Initialize attn_edge_type as a zero tensor with correct shape
        attn_edge_type = torch.zeros([N, N, 1], dtype=torch.long)

        # Initialize adj as a zero matrix of the correct shape (already done)
        adj = torch.zeros([N, N], dtype=torch.bool)

        # Set the in-degree as a zero vector
        indgree = torch.zeros(N, dtype=torch.long)

        # Set node_attr as token_type reshaped to match the expected shape
        data["node_attr"] = data["token_type"].reshape(-1, 1)

        # Initialize attn_bias with zeros and shape (N+1, N+1)
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)

        # Initialize in-degree to zero vector
        data["in_degree"] = indgree

        # Initialize shortest_path_result, path, and edge_input with zeros
        spatial_pos = torch.zeros([N, N], dtype=torch.long)
        data["edge_input"] = torch.zeros([N, N, 1], dtype=torch.long)
        data["spatial_pos"] = spatial_pos

        return data

    def build_2d_graph(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)
        if "edge_index" not in data or data["edge_index"] is None:
            data["edge_attr"] = None
            edge_index = radius_graph(data["coords"][:N], 10)
            data["edge_index"] = edge_index
        edge_index = data["edge_index"].clone().detach().to(torch.long)
        edge_attr = torch.ones((data["edge_index"].shape[1], 1), dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["token_type"].reshape(-1, 1)

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        return data

    @classmethod
    def build_graph_feature(cls, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"].clone().detach(), dtype=torch.long)
        edge_attr = torch.ones((data["edge_index"].shape[1], 1), dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["token_type"].reshape(-1, 1)

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        return data

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None

    def split_dataset(self, training_ratio=0.03, validation_ratio=0.2, sort=False):
        num_samples = self.num_samples
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)
        assert (
            training_ratio + validation_ratio <= 1.0
        ), f"Invalid training_ratio '{training_ratio}' and validation_ratio '{validation_ratio}'"

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = int(num_samples * training_ratio)

        training_indices = indices[:num_training_samples]
        validation_indices = indices[-num_validation_samples:]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        return dataset_train, dataset_val

    def split_train_valid_test(self, ratio_list: list, sort=False, shuffle=True):
        num_samples = self.num_samples

        indices = list(range(num_samples))
        # Shuffle the indices and split them into training and validation sets
        if shuffle:
            random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * ratio_list[1])
        num_test_samples = int(num_samples * ratio_list[2])
        num_training_samples = num_samples - num_validation_samples - num_test_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[
            num_training_samples : num_training_samples + num_validation_samples
        ]
        test_indices = indices[num_training_samples + num_validation_samples :]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        dataset_test = Subset(self, test_indices)
        return dataset_train, dataset_val, dataset_test


class MultiDataset:
    def __init__(
        self,
        args,
        dataset_list,
        len_data: Optional[int] = None,
        extra_collate_fn=None,
        **kwargs,
    ):
        self.args = args
        self.dataset_list = dataset_list
        self.num_datasets = len(dataset_list)
        self.extra_collate_fn = extra_collate_fn

        self.dataset_lens = [len(ds) for ds in self.dataset_list]
        self.dataset_ranges = np.cumsum([0] + self.dataset_lens).tolist()
        self.total_len = int(self.dataset_ranges[-1])

        if len_data is not None and int(len_data) != self.total_len:
            raise ValueError(
                f"len_data ({len_data}) must equal sum(len(ds)) ({self.total_len}) "
                "for MultiDataset."
            )

        logger.info(f"Total data Length is {self.total_len/1000/1000:0.2f}M")

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_len}")

        for i in range(self.num_datasets):
            start = self.dataset_ranges[i]
            end = self.dataset_ranges[i + 1]
            if start <= idx < end:
                return self.dataset_list[i][idx - start]

        raise RuntimeError(f"Data with index {idx} not found in any subset.")

    def collate(self, samples):
        batched_data = pack_batch(
            samples,
        )
        if self.extra_collate_fn is not None:
            batched_data = self.extra_collate_fn(samples, batched_data)
        return batched_data

    def num_tokens(self, idx: int) -> int:
        raise NotImplementedError("num_tokens is not implemented for this dataset.")


class ProportionalSampler(ABC):
    # samples data from different modalities
    def __init__(
        self,
        dataset: MultiDataset,
        dataset_split_ratios: str,
        dataset_batch_sizes: str,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.dataset_split_ratios = [
            float(ratio) for ratio in dataset_split_ratios.split(",")
        ]
        self.dataset_batch_sizes = [
            int(batch_size) for batch_size in dataset_batch_sizes.split(",")
        ]
        assert len(dataset.dataset_lens) == len(self.dataset_split_ratios) and len(
            dataset.dataset_lens
        ) == len(
            self.dataset_batch_sizes
        ), "Dataset parameters mismatched, please check data_path_list, dataset_name_list, dataset_split_raito, and dataset_micro_batch_size"
        self.dataset_ranges = np.cumsum([0] + dataset.dataset_lens)
        total_len = self.dataset_ranges[-1]
        dataset_sampled_lens = [
            total_len * ratio for ratio in self.dataset_split_ratios
        ]
        weight_dict = {}
        for i in range(len(self.dataset_ranges) - 1):
            start = self.dataset_ranges[i]
            end = self.dataset_ranges[i + 1]
            weight_dict[(start, end)] = dataset_sampled_lens[i] * 1.0 / (end - start)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.weight_dict = weight_dict
        self.dataset_sampled_len = {}

        dataset_indices_len = 0
        num_samples = 0
        for i in range(len(self.dataset_ranges) - 1):
            start = self.dataset_ranges[i]
            end = self.dataset_ranges[i + 1]
            ratio = weight_dict[(start, end)]
            sampled_len = math.ceil((end - start) * ratio)
            micro_batch_size = self.dataset_batch_sizes[i]
            sampled_len = (
                (sampled_len + micro_batch_size * num_replicas - 1)
                // (micro_batch_size * num_replicas)
                * micro_batch_size
                * num_replicas
            )
            self.dataset_sampled_len[(start, end)] = sampled_len
            dataset_indices_len += sampled_len
            num_samples += sampled_len // num_replicas

        self.dataset_indices_len = dataset_indices_len
        self.num_total_samples = num_samples
        self.num_samples = num_samples
        self.total_size = self.num_samples * self.num_replicas
        assert self.total_size == self.dataset_indices_len
        self.seed = seed
        self.shuffle = True
        self.drop_last = False
        self.num_skip_batches = None
        self.micro_batch_size = None

    def __iter__(self):
        generator = np.random.default_rng(self.epoch + self.seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(self.seed + self.epoch)
        indices = []
        for begin, end in np.sort(list(self.dataset_sampled_len.keys())):
            sampled_len = self.dataset_sampled_len[(begin, end)]
            indices_for_dataset = []
            while sampled_len > end - begin:
                indices_for_dataset.extend(
                    torch.randperm(end - begin, generator=torch_generator).numpy()
                    + begin
                )
                sampled_len -= end - begin
            indices_for_dataset.extend(
                list(generator.choice(end - begin, sampled_len, replace=False) + begin)
            )
            indices_for_dataset = list(
                torch.tensor(indices_for_dataset, dtype=torch.long)[
                    torch.randperm(len(indices_for_dataset), generator=torch_generator)
                ].numpy()
            )
            indices.extend(
                indices_for_dataset[self.rank : self.total_size : self.num_replicas]
            )
        sorted_indices = torch.randperm(
            len(indices), generator=torch_generator
        ).tolist()
        indices = np.array(indices)[np.array(sorted_indices)].tolist()

        assert len(indices) == self.num_total_samples
        self.num_samples = self.num_total_samples

        num_datasets = len(self.dataset_ranges) - 1
        split_indices = [[] for _ in range(num_datasets)]
        for index in indices:
            for tag in range(num_datasets):
                if (
                    index >= self.dataset_ranges[tag]
                    and index < self.dataset_ranges[tag + 1]
                ):
                    split_indices[tag].append(index)

        batch_seqs = [
            [
                split_indices[j][
                    i * self.dataset_batch_sizes[j] : i * self.dataset_batch_sizes[j]
                    + self.dataset_batch_sizes[j]
                ]
                for i in range(len(split_indices[j]) // self.dataset_batch_sizes[j])
            ]
            for j in range(num_datasets)
        ]

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        total_num_batches = np.sum([len(batch_seq) for batch_seq in batch_seqs])
        all_batches = []
        for batch_seq in batch_seqs:
            all_batches += batch_seq
        batch_indices = torch.randperm(total_num_batches, generator=g).tolist()

        if self.num_skip_batches is not None:
            batch_indices = batch_indices[self.num_skip_batches :]
        all_batches = [all_batches[i] for i in batch_indices]
        if self.num_skip_batches is not None:
            self.num_samples = np.sum(len(batch) for batch in all_batches)
        return iter(all_batches)

    def __len__(self) -> int:
        return self.num_samples

    def set_skip_batches(self, num_skip_batches, micro_batch_size):
        self.num_skip_batches = num_skip_batches
        self.micro_batch_size = micro_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
