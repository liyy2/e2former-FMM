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
            "Validate masking semantics for AlphaFRYSphericalFMM and "
            "AlphaFRYSphericalSoftmax."
        )
    )
    parser.add_argument(
        "--max-nodes",
        "--num-nodes",
        dest="max_nodes",
        type=int,
        default=80,
        help="Padded node length per graph.",
    )
    parser.add_argument("--heads", type=int, default=4, help="Number of heads.")
    parser.add_argument("--head-dim", type=int, default=16, help="Per-head q/k dim.")
    parser.add_argument("--value-dim", type=int, default=8, help="Per-head value dim.")
    parser.add_argument("--degree", type=int, default=1, help="Spherical harmonic degree l.")
    parser.add_argument("--kappa", type=int, default=8, help="Number of radial frequencies.")
    parser.add_argument("--dirs", type=int, default=25, help="FMM sphere quadrature directions.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--tol-fmm",
        type=float,
        default=5e-5,
        help="Tolerance for FMM checks.",
    )
    parser.add_argument(
        "--tol-softmax",
        type=float,
        default=1e-6,
        help="Tolerance for softmax checks.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def _max_abs_masked(x: torch.Tensor, mask: torch.Tensor) -> float:
    if bool(mask.any()):
        return float(x[mask].abs().max().item())
    return 0.0


@torch.no_grad()
def _check_masking_behavior(
    name: str,
    core: torch.nn.Module,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    tol: float,
) -> tuple[float, float, float]:
    out_ref = core(pos, q, k, v, node_mask=node_mask)

    # 1) Mutating padded nodes must not change valid outputs.
    pos_mut = pos.clone()
    q_mut = q.clone()
    k_mut = k.clone()
    v_mut = v.clone()
    pad_mask = ~node_mask
    pos_mut[pad_mask] = 1000.0 * torch.randn_like(pos_mut[pad_mask]) + 777.0
    q_mut[pad_mask] = 1000.0 * torch.randn_like(q_mut[pad_mask]) - 333.0
    k_mut[pad_mask] = 1000.0 * torch.randn_like(k_mut[pad_mask]) + 111.0
    v_mut[pad_mask] = 1000.0 * torch.randn_like(v_mut[pad_mask]) - 999.0
    out_mut = core(pos_mut, q_mut, k_mut, v_mut, node_mask=node_mask)

    valid_mask = node_mask[:, :, None, None, None].expand_as(out_ref)
    invariant_err = _max_abs_masked(out_ref - out_mut, valid_mask)

    # 2) Masked query positions should produce (near-)zero outputs.
    masked_mask = pad_mask[:, :, None, None, None].expand_as(out_ref)
    masked_out_mag = _max_abs_masked(out_ref, masked_mask)

    # 3) All-true mask should match node_mask=None.
    full_mask = torch.ones_like(node_mask)
    out_full_mask = core(pos, q, k, v, node_mask=full_mask)
    out_no_mask = core(pos, q, k, v, node_mask=None)
    fullmask_equiv_err = float((out_full_mask - out_no_mask).abs().max().item())

    print(f"{name}:")
    print(f"  invariant_err      = {invariant_err:.6e}")
    print(f"  masked_out_mag     = {masked_out_mag:.6e}")
    print(f"  fullmask_equiv_err = {fullmask_equiv_err:.6e}")

    failed = (
        invariant_err > tol
        or masked_out_mag > tol
        or fullmask_equiv_err > tol
    )
    if failed:
        raise RuntimeError(
            f"{name} masking check failed: "
            f"invariant={invariant_err:.6e}, "
            f"masked={masked_out_mag:.6e}, "
            f"fullmask={fullmask_equiv_err:.6e}, tol={tol:.6e}"
        )

    return invariant_err, masked_out_mag, fullmask_equiv_err


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    from molfm.models.e2former.fmm_prototype import (  # noqa: WPS433
        AlphaFRYSphericalFMM,
        AlphaFRYSphericalSoftmax,
    )

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
    Q = int(args.kappa)

    # Long + short graph in one padded batch.
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=device)
    node_mask[0, : max(2, max_nodes - 8)] = True
    node_mask[1, : max(2, max_nodes // 8 + 1)] = True

    pos = torch.randn(B, max_nodes, 3, device=device)
    q = torch.randn(B, max_nodes, H, D, device=device)
    k = torch.randn(B, max_nodes, H, D, device=device)
    v = torch.randn(B, max_nodes, H, C, device=device)

    kappa = torch.linspace(1.0, float(Q), Q, dtype=torch.float32)
    a = torch.randn(H, Q, dtype=torch.float32)

    fmm_core = AlphaFRYSphericalFMM(
        l=int(args.degree),
        kappa=kappa,
        a=a,
        num_directions=int(args.dirs),
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(device)
    softmax_core = AlphaFRYSphericalSoftmax(
        l=int(args.degree),
        kappa=kappa,
        a=a,
    ).to(device)

    fmm_core.eval()
    softmax_core.eval()

    print("== Masking behavior checks ==")
    print(
        f"device={device} B={B} max_nodes={max_nodes} valid_lengths="
        f"[{int(node_mask[0].sum().item())}, {int(node_mask[1].sum().item())}] "
        f"heads={H} head_dim={D} value_dim={C} l={int(args.degree)}"
    )
    _check_masking_behavior(
        name="FMM",
        core=fmm_core,
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        tol=float(args.tol_fmm),
    )
    _check_masking_behavior(
        name="Softmax",
        core=softmax_core,
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        tol=float(args.tol_softmax),
    )


if __name__ == "__main__":
    main()
