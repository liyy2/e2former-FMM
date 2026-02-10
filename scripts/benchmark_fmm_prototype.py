# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
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
    p = argparse.ArgumentParser(
        description="Benchmark softmax vs vanilla(linear) vs FMM prototype (alpha * f(r) * Y)."
    )
    p.add_argument("--N", type=int, default=None, help="Total nodes (ignored if --B and --L are set)")
    p.add_argument("--B", type=int, default=None, help="Batch size for E2Former-like shapes")
    p.add_argument("--L", type=int, default=None, help="Nodes per batch for E2Former-like shapes")
    p.add_argument("--heads", type=int, default=1, help="Number of heads (q/k/v will have an extra head dim)")
    p.add_argument("--d", type=int, default=64, help="Head dimension for q/k (or feature dim if heads=1)")
    p.add_argument("--C", type=int, default=32, help="Per-head value dim")
    p.add_argument("--l", type=int, default=2)
    p.add_argument("--Q", type=int, default=8)
    p.add_argument("--phase-mode", type=str, default="complex", choices=["complex", "trig"])
    p.add_argument("--kappa-chunk-size", type=int, default=0)
    p.add_argument("--compile", action="store_true", help="Wrap FMM with torch.compile.")
    p.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--no-promote-half-precision",
        action="store_true",
        help="Keep fp16/bf16 compute in FMM instead of promoting to fp32.",
    )
    p.add_argument(
        "--S",
        type=int,
        default=146,
        help="Sphere directions / quadrature points (for Lebedev, use supported counts like 74/110/146/194/302).",
    )
    p.add_argument(
        "--sphere",
        type=str,
        default="lebedev",
        choices=["lebedev", "fibonacci", "gauss_legendre", "e3nn_s2grid"],
    )
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    p.add_argument("--device", type=str, default=None, help="e.g. cpu / cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _time_forward(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> float:
    device = inputs[0].device
    module.eval()
    for _ in range(warmup):
        _ = module(*inputs)
    _maybe_sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = module(*inputs)
    _maybe_sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    from molfm.models.e2former.fmm_prototype import (  # noqa: WPS433
        AlphaFRYSphericalFMM,
        AlphaFRYSphericalSoftmax,
        AlphaFRYSphericalVanilla,
    )

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float64

    torch.manual_seed(args.seed)
    if (args.B is None) != (args.L is None):
        raise ValueError("Please provide both --B and --L, or neither.")
    if args.B is not None and args.L is not None:
        B, L = int(args.B), int(args.L)
        pos = torch.randn(B, L, 3, device=device, dtype=dtype) * args.pos_scale
        q = torch.randn(B, L, args.heads, args.d, device=device, dtype=dtype)
        k = torch.randn(B, L, args.heads, args.d, device=device, dtype=dtype)
        v = torch.randn(B, L, args.heads, args.C, device=device, dtype=dtype)
        N_report = B * L
    else:
        if args.N is None:
            args.N = 512
        pos = torch.randn(args.N, 3, device=device, dtype=dtype) * args.pos_scale
        q = torch.randn(args.N, args.heads, args.d, device=device, dtype=dtype)
        k = torch.randn(args.N, args.heads, args.d, device=device, dtype=dtype)
        v = torch.randn(args.N, args.heads, args.C, device=device, dtype=dtype)
        N_report = int(args.N)

    # Radial f(r) = sum_q a_q j_l(kappa_q r)
    kappa = torch.linspace(1.0, float(args.Q), args.Q, device=device, dtype=dtype)
    a = torch.randn(args.Q, device=device, dtype=dtype) / math.sqrt(float(args.Q))

    vanilla = AlphaFRYSphericalVanilla(args.l, kappa=kappa, a=a).to(device=device, dtype=dtype)
    softmax = AlphaFRYSphericalSoftmax(args.l, kappa=kappa, a=a).to(device=device, dtype=dtype)
    fmm_core = AlphaFRYSphericalFMM(
        args.l,
        kappa=kappa,
        a=a,
        num_directions=args.S,
        sphere=args.sphere,
        phase_mode=args.phase_mode,
        kappa_chunk_size=args.kappa_chunk_size,
        promote_half_precision=not args.no_promote_half_precision,
    ).to(device=device, dtype=dtype)
    resolved_sphere = fmm_core.sphere
    resolved_S = int(fmm_core.directions.shape[0])
    fmm = fmm_core
    if args.compile:
        fmm = torch.compile(fmm, mode=args.compile_mode)

    inputs = (pos, q, k, v)

    t_vanilla = _time_forward(vanilla, inputs, warmup=args.warmup, iters=args.iters)
    t_softmax = _time_forward(softmax, inputs, warmup=args.warmup, iters=args.iters)
    t_fmm = _time_forward(fmm, inputs, warmup=args.warmup, iters=args.iters)

    with torch.no_grad():
        out_vanilla = vanilla(*inputs)
        out_softmax = softmax(*inputs)
        out_fmm = fmm(*inputs)

    metric_dtype = torch.float64 if out_vanilla.dtype == torch.float64 else torch.float32
    out_vanilla_m = out_vanilla.to(dtype=metric_dtype)
    out_softmax_m = out_softmax.to(dtype=metric_dtype)
    out_fmm_m = out_fmm.to(dtype=metric_dtype)

    diff_soft_van = out_softmax_m - out_vanilla_m
    rel_l2_soft_van = diff_soft_van.norm() / out_softmax_m.norm().clamp_min(1e-12)
    max_abs_soft_van = diff_soft_van.abs().max()

    diff_fmm_van = out_fmm_m - out_vanilla_m
    rel_l2_fmm_van = diff_fmm_van.norm() / out_vanilla_m.norm().clamp_min(1e-12)
    max_abs_fmm_van = diff_fmm_van.abs().max()

    diff_fmm_softmax = out_fmm_m - out_softmax_m
    rel_l2_fmm_softmax = diff_fmm_softmax.norm() / out_softmax_m.norm().clamp_min(1e-12)
    max_abs_fmm_softmax = diff_fmm_softmax.abs().max()

    print("== Prototype benchmark: alpha * f(r) * Y ==")
    print(
        f"device={device} dtype={dtype} N={N_report} heads={args.heads} d={args.d} C={args.C} "
        f"l={args.l} Q={args.Q} S_req={args.S} S_used={resolved_S} "
        f"sphere_req={args.sphere} sphere_used={resolved_sphere} phase_mode={args.phase_mode} "
        f"kappa_chunk_size={args.kappa_chunk_size} compile={args.compile} "
        f"promote_half_precision={not args.no_promote_half_precision}"
    )
    print(f"vanilla_forward: {t_vanilla * 1e3:.3f} ms")
    print(f"softmax_forward: {t_softmax * 1e3:.3f} ms")
    print(f"fmm_forward:     {t_fmm * 1e3:.3f} ms")
    print(f"speedup van/fmm: {t_vanilla / max(t_fmm, 1e-12):.2f}x")
    print(f"rel_l2 soft-van: {rel_l2_soft_van.item():.4e} (max_abs {max_abs_soft_van.item():.4e})")
    print(
        f"rel_l2 fmm-van:  {rel_l2_fmm_van.item():.4e} (max_abs {max_abs_fmm_van.item():.4e})"
    )
    print(
        f"rel_l2 fmm-soft: {rel_l2_fmm_softmax.item():.4e} (max_abs {max_abs_fmm_softmax.item():.4e})"
    )


if __name__ == "__main__":
    main()
