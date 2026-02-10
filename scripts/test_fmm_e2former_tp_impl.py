# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from e3nn import o3


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the combined-e3nn-TP implementation in E2AttentionNodeFMM: "
            "coupling equivalence (TP vs explicit Wigner-3j), node permutation "
            "equivariance, and graph isolation in multi-graph batches."
        )
    )
    parser.add_argument(
        "--irreps-node-input",
        type=str,
        default="8x0e+4x1e+2x2e",
        help="Input irreps for E2AttentionNodeFMM.",
    )
    parser.add_argument("--heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--head-dim", type=int, default=4, help="Per-head q/k dim.")
    parser.add_argument(
        "--graph-lengths",
        type=str,
        default="11,6,3",
        help="Comma-separated node counts for graphs in the packed batch.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--tp-backend",
        type=str,
        default="cueq",
        choices=["auto", "e3nn", "cueq"],
        help="Requested tensor-product backend for E2AttentionNodeFMM.",
    )
    parser.add_argument(
        "--tol-coupling",
        type=float,
        default=2e-4,
        help="Tolerance for TP-vs-explicit coupling equivalence.",
    )
    parser.add_argument(
        "--tol-perm",
        type=float,
        default=2e-4,
        help="Tolerance for node permutation equivariance.",
    )
    parser.add_argument(
        "--tol-iso",
        type=float,
        default=2e-4,
        help="Tolerance for cross-graph isolation check.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def _parse_graph_lengths(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError(f"Need >=2 graphs for isolation checks, got: {values}")
    if any(length <= 0 for length in values):
        raise ValueError(f"Graph lengths must be positive, got: {values}")
    return values


def _make_batch(lengths: list[int], device: torch.device) -> torch.Tensor:
    parts = [
        torch.full((length,), graph_id, dtype=torch.long, device=device)
        for graph_id, length in enumerate(lengths)
    ]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _compute_fmm_by_ell(
    module,
    packed_pos: torch.Tensor,
    packed_irreps: torch.Tensor,
    packed_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    num_graphs, max_nodes = packed_pos.shape[:2]
    num_irrep_components = module.num_irrep_components

    packed_irreps_f = packed_irreps.to(dtype=torch.float32)
    q = module.q_proj(packed_irreps_f).view(
        num_graphs, max_nodes, module.num_attn_heads, module.attn_scalar_head
    )
    k = module.k_proj(packed_irreps_f).view(
        num_graphs, max_nodes, module.num_attn_heads, module.attn_scalar_head
    )

    v_all = packed_irreps.view(
        num_graphs,
        max_nodes,
        num_irrep_components,
        module.num_attn_heads,
        module.value_head_dim,
    ).permute(0, 1, 3, 2, 4).contiguous()
    v_all_flat = v_all.view(
        num_graphs,
        max_nodes,
        module.num_attn_heads,
        num_irrep_components * module.value_head_dim,
    )

    raw_out_by_ell = module.fmm_multi_l(
        packed_pos,
        q,
        k,
        v_all_flat,
        node_mask=packed_mask,
    )
    out_by_ell: list[torch.Tensor] = []
    for ell, out_ell in enumerate(raw_out_by_ell):
        m_ell = 2 * ell + 1
        out_by_ell.append(
            out_ell.view(
                num_graphs,
                max_nodes,
                module.num_attn_heads,
                m_ell,
                num_irrep_components,
                module.value_head_dim,
            )
        )
    return v_all_flat, out_by_ell


@torch.no_grad()
def _couple_with_combined_tp(module, out_by_ell: list[torch.Tensor]) -> torch.Tensor:
    ref_tensor = out_by_ell[0]
    num_graphs, max_nodes = ref_tensor.shape[:2]
    packed_out = ref_tensor.new_zeros(
        (num_graphs, max_nodes, module.num_irrep_components, module.scalar_dim)
    )

    coupled_all = None
    for ell, out_ell_all in enumerate(out_by_ell):
        ell_start, ell_end = module._l_slice(ell)
        tp_right_slice = module.tp_right_all[:, ell_start:ell_end, :].to(
            device=out_ell_all.device,
            dtype=out_ell_all.dtype,
        )
        contrib = torch.einsum("bnhpqc,qpr->bnhrc", out_ell_all, tp_right_slice)
        if coupled_all is None:
            coupled_all = contrib
        else:
            coupled_all.add_(contrib)

    if coupled_all is None:
        raise RuntimeError("No ell blocks were generated for TP coupling.")

    for out_l, start, end, mul in module.tp_out_blocks:
        m_out = 2 * out_l + 1
        block = coupled_all[:, :, :, start:end, :]
        if mul > 1:
            block = block.view(
                num_graphs,
                max_nodes,
                module.num_attn_heads,
                mul,
                m_out,
                module.value_head_dim,
            ).sum(dim=3)
        else:
            block = block.view(
                num_graphs,
                max_nodes,
                module.num_attn_heads,
                m_out,
                module.value_head_dim,
            )
        block = block.permute(0, 1, 3, 2, 4).contiguous()
        block = block.view(num_graphs, max_nodes, m_out, module.scalar_dim)
        out_start, out_end = module._l_slice(out_l)
        packed_out[:, :, out_start:out_end, :].add_(block)

    for out_l in range(module.lmax + 1):
        out_start, out_end = module._l_slice(out_l)
        packed_out[:, :, out_start:out_end, :].div_(module.coupling_count[out_l])
    return packed_out


@torch.no_grad()
def _couple_with_explicit_cg(module, out_by_ell: list[torch.Tensor]) -> torch.Tensor:
    ref_tensor = out_by_ell[0]
    num_graphs, max_nodes = ref_tensor.shape[:2]
    packed_out = ref_tensor.new_zeros(
        (num_graphs, max_nodes, module.num_irrep_components, module.scalar_dim)
    )
    coupling_count = torch.zeros(
        module.lmax + 1, device=ref_tensor.device, dtype=ref_tensor.dtype
    )

    for ell, out_ell_all in enumerate(out_by_ell):
        for lam in range(module.lmax + 1):
            lam_start, lam_end = module._l_slice(lam)
            out_ell_lam = out_ell_all[:, :, :, :, lam_start:lam_end, :]
            l_min = abs(ell - lam)
            l_max = min(ell + lam, module.lmax)
            for out_l in range(l_min, l_max + 1):
                cg = o3.wigner_3j(ell, lam, out_l).to(
                    device=out_ell_lam.device, dtype=out_ell_lam.dtype
                )
                coupled = torch.einsum("pqr,bnhpqc->bnhrc", cg, out_ell_lam)
                coupled = coupled.permute(0, 1, 3, 2, 4).contiguous()
                coupled = coupled.view(
                    num_graphs,
                    max_nodes,
                    2 * out_l + 1,
                    module.scalar_dim,
                )
                out_start, out_end = module._l_slice(out_l)
                packed_out[:, :, out_start:out_end, :].add_(coupled)
                coupling_count[out_l] += 1.0

    coupling_count = coupling_count.clamp_min(1.0)
    for out_l in range(module.lmax + 1):
        out_start, out_end = module._l_slice(out_l)
        packed_out[:, :, out_start:out_end, :].div_(coupling_count[out_l])
    return packed_out


@torch.no_grad()
def _check_coupling_equivalence(
    module,
    node_pos: torch.Tensor,
    node_irreps: torch.Tensor,
    batch: torch.Tensor,
    tol: float,
) -> float:
    packed_pos, packed_irreps, packed_mask, _ = module._pack_by_graph(
        node_pos=node_pos,
        node_irreps=node_irreps,
        batch=batch,
    )
    _, out_by_ell = _compute_fmm_by_ell(
        module=module,
        packed_pos=packed_pos,
        packed_irreps=packed_irreps,
        packed_mask=packed_mask,
    )
    out_tp = _couple_with_combined_tp(module, out_by_ell)
    out_cg = _couple_with_explicit_cg(module, out_by_ell)
    err = float((out_tp - out_cg).abs().max().item())
    if err > tol:
        raise RuntimeError(f"coupling_equiv_err={err:.6e} exceeds tol={tol:.6e}")
    return err


@torch.no_grad()
def _check_permutation_equivariance(
    module,
    node_pos: torch.Tensor,
    node_irreps: torch.Tensor,
    batch: torch.Tensor,
    attn_weight: torch.Tensor,
    tol: float,
) -> float:
    out_ref, _ = module(
        node_pos=node_pos,
        node_irreps_input=node_irreps,
        edge_dis=None,
        edge_vec=None,
        attn_weight=attn_weight,
        atomic_numbers=None,
        batch=batch,
    )
    perm = torch.randperm(node_pos.shape[0], device=node_pos.device)
    inv_perm = torch.argsort(perm)
    out_perm, _ = module(
        node_pos=node_pos[perm],
        node_irreps_input=node_irreps[perm],
        edge_dis=None,
        edge_vec=None,
        attn_weight=attn_weight[perm],
        atomic_numbers=None,
        batch=batch[perm],
    )
    err = float((out_ref - out_perm[inv_perm]).abs().max().item())
    if err > tol:
        raise RuntimeError(f"permutation_err={err:.6e} exceeds tol={tol:.6e}")
    return err


@torch.no_grad()
def _check_graph_isolation(
    module,
    node_pos: torch.Tensor,
    node_irreps: torch.Tensor,
    batch: torch.Tensor,
    attn_weight: torch.Tensor,
    tol: float,
) -> float:
    out_ref, _ = module(
        node_pos=node_pos,
        node_irreps_input=node_irreps,
        edge_dis=None,
        edge_vec=None,
        attn_weight=attn_weight,
        atomic_numbers=None,
        batch=batch,
    )

    graph_ids = torch.unique(batch)
    if graph_ids.numel() < 2:
        raise RuntimeError("Need >=2 graphs for isolation check.")
    target_graph = int(graph_ids.max().item())
    source_graph = int(graph_ids.min().item())

    pos_mut = node_pos.clone()
    irreps_mut = node_irreps.clone()
    target_mask = batch == target_graph
    source_mask = batch == source_graph

    pos_mut[target_mask] = pos_mut[target_mask] + 100.0
    irreps_mut[target_mask] = irreps_mut[target_mask] * 9.0 + 7.0

    out_mut, _ = module(
        node_pos=pos_mut,
        node_irreps_input=irreps_mut,
        edge_dis=None,
        edge_vec=None,
        attn_weight=attn_weight,
        atomic_numbers=None,
        batch=batch,
    )
    err = float((out_ref[source_mask] - out_mut[source_mask]).abs().max().item())
    if err > tol:
        raise RuntimeError(f"isolation_err={err:.6e} exceeds tol={tol:.6e}")
    return err


def main() -> None:
    args = _parse_args()
    _add_src_to_path()
    from molfm.models.e2former.fmm_e2former import E2AttentionNodeFMM  # noqa: WPS433

    torch.manual_seed(int(args.seed))
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    graph_lengths = _parse_graph_lengths(args.graph_lengths)
    batch = _make_batch(graph_lengths, device=device)
    n_nodes = int(batch.shape[0])

    if args.tp_backend == "cueq":
        tp_type = "fmm-node+tp_cueq"
    elif args.tp_backend == "e3nn":
        tp_type = "fmm-node+tp_e3nn"
    else:
        tp_type = "fmm-node"

    module = E2AttentionNodeFMM(
        irreps_node_input=args.irreps_node_input,
        num_attn_heads=int(args.heads),
        attn_scalar_head=int(args.head_dim),
        tp_type=tp_type,
    ).to(device)
    module.eval()
    if args.tp_backend != "auto" and module.tp_backend != args.tp_backend:
        raise RuntimeError(
            f"Requested tp_backend={args.tp_backend} but resolved tp_backend={module.tp_backend}"
        )

    node_pos = torch.randn(n_nodes, 3, device=device)
    node_irreps = torch.randn(
        n_nodes,
        module.num_irrep_components,
        module.scalar_dim,
        device=device,
    )
    attn_weight = torch.zeros(n_nodes, 1, device=device)

    coupling_err = _check_coupling_equivalence(
        module=module,
        node_pos=node_pos,
        node_irreps=node_irreps,
        batch=batch,
        tol=float(args.tol_coupling),
    )
    perm_err = _check_permutation_equivariance(
        module=module,
        node_pos=node_pos,
        node_irreps=node_irreps,
        batch=batch,
        attn_weight=attn_weight,
        tol=float(args.tol_perm),
    )
    iso_err = _check_graph_isolation(
        module=module,
        node_pos=node_pos,
        node_irreps=node_irreps,
        batch=batch,
        attn_weight=attn_weight,
        tol=float(args.tol_iso),
    )

    print("== E2AttentionNodeFMM combined-TP checks ==")
    print(
        f"device={device} n_nodes={n_nodes} graph_lengths={graph_lengths} "
        f"lmax={module.lmax} heads={module.num_attn_heads} scalar_dim={module.scalar_dim} "
        f"tp_backend={module.tp_backend}"
    )
    print(f"coupling_equiv_err: {coupling_err:.6e}")
    print(f"permutation_err:    {perm_err:.6e}")
    print(f"isolation_err:      {iso_err:.6e}")


if __name__ == "__main__":
    main()
