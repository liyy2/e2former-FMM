# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark baseline edge-based E2former attention against "
            "the node-only FMM-backed variant."
        )
    )
    parser.add_argument("--B", type=int, default=2, help="Batch size.")
    parser.add_argument(
        "--nodes-per-graph",
        "--num-nodes",
        dest="nodes_per_graph",
        type=int,
        default=512,
        help="Number of nodes per graph.",
    )
    parser.add_argument("--layers", type=int, default=2, help="Number of Transformer blocks.")
    parser.add_argument("--max-neighbors", type=int, default=512, help="Neighbor cap for baseline.")
    parser.add_argument(
        "--radius",
        type=float,
        default=100.0,
        help="Cutoff radius used by baseline (set large for dense-neighbor stress test).",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=0.1,
        help="Scale of random positions. Smaller values increase baseline neighborhood density.",
    )
    parser.add_argument("--number-of-basis", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attn-scalar-head", type=int, default=32)
    parser.add_argument(
        "--fmm-tp-backend",
        type=str,
        default="cueq",
        choices=["auto", "e3nn", "cueq"],
        help="Tensor-product backend for fmm-node attention.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _time_forward(
    model: torch.nn.Module,
    batched_data: dict[str, torch.Tensor],
    padding_mask: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    for _ in range(warmup):
        _ = model(batched_data=batched_data, token_embedding=None, padding_mask=padding_mask)
    _maybe_sync(padding_mask.device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(batched_data=batched_data, token_embedding=None, padding_mask=padding_mask)
    _maybe_sync(padding_mask.device)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    from molfm.models.e2former.e2former import E2former  # noqa: WPS433

    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    nodes_per_graph = int(args.nodes_per_graph)

    # Keep a known-valid E2former setting for both variants.
    common = dict(
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=args.layers,
        pbc_max_radius=float(args.radius),
        max_radius=float(args.radius),
        basis_type="gaussiansmear",
        number_of_basis=args.number_of_basis,
        num_attn_heads=args.heads,
        attn_scalar_head=args.attn_scalar_head,
        irreps_head="32x0e+32x1e+32x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="rms_norm_sh_BL",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        atom_type_cnt=256,
        edge_embedtype="eqv2",
        attn_biastype="share",
        ffn_type="eqv2ffn",
        add_rope=False,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        force_head=None,
        decouple_EF=False,
        max_neighbors=args.max_neighbors,
    )

    pos = args.pos_scale * torch.randn(args.B, nodes_per_graph, 3, device=device)
    masked_token_type = torch.randint(1, 20, (args.B, nodes_per_graph), device=device)
    padding_mask = torch.zeros(args.B, nodes_per_graph, dtype=torch.bool, device=device)
    pbc = torch.zeros(args.B, 3, dtype=torch.bool, device=device)
    batched_data = {
        "pos": pos,
        "masked_token_type": masked_token_type,
        "pbc": pbc,
    }

    baseline = E2former(
        tp_type="QK_alpha",
        attn_type="first-order",
        **common,
    ).to(device)
    if args.fmm_tp_backend == "e3nn":
        fmm_tp_type = "fmm-node+tp_e3nn"
    elif args.fmm_tp_backend == "cueq":
        fmm_tp_type = "fmm-node+tp_cueq"
    else:
        fmm_tp_type = "fmm-node"
    fmm_node = E2former(
        tp_type=fmm_tp_type,
        attn_type="fmm-node",
        **common,
    ).to(device)

    t_baseline = _time_forward(
        baseline, batched_data, padding_mask, warmup=args.warmup, iters=args.iters
    )
    t_fmm_node = _time_forward(
        fmm_node, batched_data, padding_mask, warmup=args.warmup, iters=args.iters
    )

    print("== E2former attention benchmark ==")
    print(
        f"device={device} B={args.B} num_nodes={nodes_per_graph} layers={args.layers} "
        f"max_neighbors={args.max_neighbors} radius={args.radius} pos_scale={args.pos_scale} "
        f"fmm_tp_backend={args.fmm_tp_backend}"
    )
    print(f"baseline(edge) forward: {t_baseline * 1e3:.3f} ms")
    print(f"fmm-node forward:       {t_fmm_node * 1e3:.3f} ms")
    print(f"speedup baseline/fmm:   {t_baseline / max(t_fmm_node, 1e-12):.2f}x")


if __name__ == "__main__":
    main()
