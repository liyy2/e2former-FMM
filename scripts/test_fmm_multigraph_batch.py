# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
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
            "Check multi-graph correctness for AlphaFRYSphericalFMM. "
            "Runs both permutation and per-graph isolation checks."
        )
    )
    parser.add_argument(
        "--max-nodes",
        "--num-nodes",
        dest="max_nodes",
        type=int,
        default=16,
        help="Max padded nodes per graph.",
    )
    parser.add_argument("--heads", type=int, default=4, help="Number of heads.")
    parser.add_argument("--head-dim", type=int, default=8, help="Per-head q/k dim.")
    parser.add_argument("--value-dim", type=int, default=8, help="Per-head value dim.")
    parser.add_argument("--degree", type=int, default=1, help="Spherical harmonic degree l.")
    parser.add_argument("--kappa", type=int, default=8, help="Number of radial frequencies.")
    parser.add_argument("--dirs", type=int, default=25, help="Sphere quadrature directions.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tol", type=float, default=5e-5, help="Absolute tolerance.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def _valid_max_abs(x: torch.Tensor, mask: torch.Tensor) -> float:
    if bool(mask.any()):
        return float(x[mask].abs().max().item())
    return 0.0


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    B = 2
    max_nodes = int(args.max_nodes)
    H = int(args.heads)
    D = int(args.head_dim)
    C = int(args.value_dim)

    kappa = torch.linspace(1.0, float(args.kappa), int(args.kappa), dtype=torch.float32)
    a = torch.randn(H, int(args.kappa), dtype=torch.float32)

    core = AlphaFRYSphericalFMM(
        l=int(args.degree),
        kappa=kappa,
        a=a,
        num_directions=int(args.dirs),
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(device)
    core.eval()

    # Two different graphs packed into one batch with different valid lengths.
    pos = torch.randn(B, max_nodes, 3, device=device)
    q = torch.randn(B, max_nodes, H, D, device=device)
    k = torch.randn(B, max_nodes, H, D, device=device)
    v = torch.randn(B, max_nodes, H, C, device=device)
    mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=device)
    mask[0, : max(1, max_nodes - 3)] = True
    mask[1, : max(1, max_nodes - 7)] = True

    with torch.no_grad():
        out_batch = core(pos, q, k, v, node_mask=mask)  # (B, N_nodes, H, M, C)

        # Test 1: batch-order permutation equivariance.
        perm = torch.tensor([1, 0], device=device)
        inv_perm = torch.argsort(perm)
        out_perm = core(
            pos[perm],
            q[perm],
            k[perm],
            v[perm],
            node_mask=mask[perm],
        )
        out_perm_recovered = out_perm[inv_perm]
        perm_err = float((out_batch - out_perm_recovered).abs().max().item())

        # Test 2: per-graph isolation (no cross-graph interaction).
        out_graph0_solo = core(pos[:1], q[:1], k[:1], v[:1], node_mask=mask[:1])[0]
        out_graph1_solo = core(pos[1:2], q[1:2], k[1:2], v[1:2], node_mask=mask[1:2])[0]
        graph0_err = _valid_max_abs(out_batch[0] - out_graph0_solo, mask[0])
        graph1_err = _valid_max_abs(out_batch[1] - out_graph1_solo, mask[1])

    print("== AlphaFRYSphericalFMM multi-graph checks ==")
    print(
        f"device={device} B={B} max_nodes={max_nodes} heads={H} "
        f"head_dim={D} value_dim={C} l={args.degree}"
    )
    print(f"permutation max|diff|: {perm_err:.6e}")
    print(f"graph-0 isolation max|diff|: {graph0_err:.6e}")
    print(f"graph-1 isolation max|diff|: {graph1_err:.6e}")

    failed = any(err > float(args.tol) for err in (perm_err, graph0_err, graph1_err))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
