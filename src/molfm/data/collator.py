# -*- coding: utf-8 -*-
import torch


def _pad_1d(x, padlen, pad_value=0):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen], pad_value, dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_2d(x, padlen, pad_value=0):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_full([padlen, xdim], pad_value, dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_square(x, padlen, pad_value=0):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen, padlen], pad_value, dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_attn_edge_input(x, padlen):
    x = x + 1
    xlen = x.size(0)
    num_features = x.size(-1)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, num_features], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_3d(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_pos(x, padlen):
    return _pad_2d(x, padlen, pad_value=0)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(
        0, feature_num * offset, offset, dtype=torch.long, device=x.device
    )
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    return _pad_1d(x, padlen, pad_value=0)


def pad_cluster_ids(x, padlen):
    return _pad_1d(x, padlen, pad_value=-1)


def pad_2d_unsqueeze(x, padlen):
    return _pad_2d(x, padlen, pad_value=0)


def pad_attn_bias_unsqueeze(x, padlen):
    return _pad_square(x, padlen, pad_value=0.0)


def pad_attn_edge_input_unsqueeze(x, padlen):
    return _pad_attn_edge_input(x, padlen)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    return _pad_square(x, padlen, pad_value=0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    return _pad_3d(x, padlen1, padlen2, padlen3)


def pad_pos_unsqueeze(x, padlen):
    return _pad_pos(x, padlen)


def pad_edge_info_unsqueeze(x, padlen):
    return _pad_2d(x, padlen, pad_value=0)


def pack_batch(
    items,
    use_pbc=True,
    sample_in_validation: bool = False,
):
    _ensure_defaults(items)

    max_node_num = max(i["token_type"].shape[0] for i in items)
    max_cluster_num = (
        max(i["cluster_centers"].shape[0] for i in items)
        if "cluster_centers" in items[0]
        else 0
    )

    idx = torch.tensor([i["idx"] for i in items], dtype=torch.long)
    energy = torch.cat([i["energy"] for i in items])
    energy_per_atom = torch.cat([i["energy_per_atom"] for i in items])
    forces = torch.cat([_pad_pos(i["forces"], max_node_num) for i in items])
    x = torch.cat([_pad_2d(i["node_attr"], max_node_num) for i in items])
    attn_bias = torch.cat(
        [_pad_square(i["attn_bias"], max_node_num + 1, pad_value=0.0) for i in items]
    )
    pos = torch.cat([_pad_pos(i["coords"], max_node_num) for i in items])

    has_energy = torch.cat([i["has_energy"] for i in items], dim=0)
    has_forces = torch.cat([i["has_forces"] for i in items], dim=0)
    charge = _collect_charge(items)

    cluster_ids = None
    cluster_centers = None
    if "cluster_ids" in items[0]:
        cluster_ids = torch.cat(
            [_pad_1d(i["cluster_ids"].long(), max_node_num, pad_value=-1) for i in items]
        )
    if "cluster_centers" in items[0]:
        cluster_centers = torch.cat(
            [_pad_pos(i["cluster_centers"], max_cluster_num) for i in items]
        )

    pbc = torch.cat([i["pbc"].unsqueeze(0) for i in items], dim=0) if use_pbc else None
    cell = torch.cat([i["cell"].unsqueeze(0) for i in items], dim=0) if use_pbc else None
    num_atoms = torch.tensor([i["num_atoms"] for i in items]) if use_pbc else None

    node_type_edge = _build_node_type_edges(items, max_node_num)

    batched_data = {
        "idx": idx,
        "attn_bias": attn_bias,
        "token_id": x[:, :, 0],
        "node_attr": x,
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "forces": forces,
        "has_energy": has_energy,
        "has_forces": has_forces,
        "pos": pos,
        "node_type_edge": node_type_edge,
        "pbc": pbc,
        "cell": cell,
        "num_atoms": num_atoms,
        "charge": charge,
    }

    if sample_in_validation and "edge_attr" in items[0] and items[0]["edge_attr"] is not None:
        batched_data.update(_build_edge_restore(items))
        if "key" in items[0]:
            batched_data["key"] = [i["key"] for i in items]

    if "data_name" in items[0]:
        batched_data["data_name"] = [i["data_name"] for i in items]
    if cluster_ids is not None:
        batched_data["cluster_ids"] = cluster_ids
    if cluster_centers is not None:
        batched_data["cluster_centers"] = cluster_centers
    return batched_data


def _ensure_defaults(items):
    for item in items:
        if "pbc" not in item:
            item["pbc"] = torch.tensor([False, False, False])
        if "cell" not in item:
            item["cell"] = torch.zeros([3, 3])
        if "num_atoms" not in item:
            item["num_atoms"] = item["x"].size()[0]


def _collect_charge(items):
    if "charge" not in items[0]:
        return None
    charge = torch.Tensor([i["charge"] for i in items])
    return charge.reshape(-1, 1).long()


def _build_edge_restore(items):
    max_num_edges = max(i["edge_attr"].size()[0] for i in items)
    edge_attr = torch.cat([_pad_2d(i["edge_attr"], max_num_edges) for i in items])
    edge_index = torch.cat(
        [_pad_2d(i["edge_index"].T, max_num_edges) for i in items]
    )
    num_edges = torch.tensor(
        [int(i["edge_attr"].size()[0]) for i in items], dtype=torch.long
    )
    idx = torch.tensor([int(i["idx"]) for i in items], dtype=torch.long)
    return {
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "num_edges": num_edges,
        "idx": idx,
    }


def _build_node_type_edges(items, max_node_num):
    node_type_edges = []
    for item in items:
        node_atom_type = item["token_type"]
        n_nodes = node_atom_type.shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes) + 1
        node_atom_i = _pad_square(node_atom_i, max_node_num, pad_value=0).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1) + 1
        node_atom_j = _pad_square(node_atom_j, max_node_num, pad_value=0).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())
    return torch.cat(node_type_edges)
