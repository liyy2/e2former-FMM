# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_int_list(x: str) -> list[int]:
    items = [s.strip() for s in x.split(",") if s.strip()]
    return [int(s) for s in items]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU sweep benchmark for softmax vs vanilla(linear) vs FMM prototype (alpha * f(r) * Y)."
    )
    p.add_argument("--Ns", type=_parse_int_list, default=_parse_int_list("512,1024,2048"))
    p.add_argument(
        "--S-values",
        type=_parse_int_list,
        default=_parse_int_list("50,74,110,146,194,302"),
        help="List of sphere sizes (for Lebedev, use supported point counts).",
    )
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--C", type=int, default=32)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--l", type=int, default=2)
    p.add_argument("--Q", type=int, default=8)
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument("--sphere", type=str, default="lebedev", choices=["lebedev", "fibonacci"])
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="outputs/benchmark_fmm_prototype_gpu.png",
        help="Output plot path",
    )
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
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

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = module(*inputs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        return (ms / 1e3) / float(iters)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = module(*inputs)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(name)


@dataclass(frozen=True)
class Row:
    N: int
    S: int
    t_softmax: float
    t_vanilla: float
    t_fmm: float
    speedup_vanilla: float
    rel_l2_soft_van: float
    rel_l2_fmm_van: float
    rel_l2_fmm_soft: float
    max_abs_soft_van: float
    max_abs_fmm_van: float
    max_abs_fmm_soft: float


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    from molfm.models.e2former.fmm_prototype import (  # noqa: WPS433
        AlphaFRYSphericalFMM,
        AlphaFRYSphericalSoftmax,
        AlphaFRYSphericalVanilla,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dtype = _dtype_from_str(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Radial f(r) = sum_q a_q j_l(kappa_q r)
    kappa = torch.linspace(1.0, float(args.Q), args.Q, device=device, dtype=torch.float32)
    a = (torch.randn(args.Q, device=device, dtype=torch.float32) / math.sqrt(float(args.Q))).to(
        torch.float32
    )

    rows: list[Row] = []

    for N in args.Ns:
        pos = (torch.randn(N, 3, device=device, dtype=dtype) * args.pos_scale).contiguous()
        q = torch.randn(N, args.heads, args.d, device=device, dtype=dtype).contiguous()
        k = torch.randn(N, args.heads, args.d, device=device, dtype=dtype).contiguous()
        v = torch.randn(N, args.heads, args.C, device=device, dtype=dtype).contiguous()
        inputs = (pos, q, k, v)

        vanilla = AlphaFRYSphericalVanilla(args.l, kappa=kappa, a=a).to(device=device, dtype=dtype)
        softmax = AlphaFRYSphericalSoftmax(args.l, kappa=kappa, a=a).to(device=device, dtype=dtype)
        t_vanilla = _time_forward(vanilla, inputs, warmup=args.warmup, iters=args.iters)
        t_softmax = _time_forward(softmax, inputs, warmup=args.warmup, iters=args.iters)

        with torch.no_grad():
            out_vanilla = vanilla(*inputs)
            out_softmax = softmax(*inputs)

        metric_dtype = torch.float64 if out_vanilla.dtype == torch.float64 else torch.float32
        out_vanilla_m = out_vanilla.to(dtype=metric_dtype)
        out_softmax_m = out_softmax.to(dtype=metric_dtype)

        diff_soft_van = out_softmax_m - out_vanilla_m
        rel_l2_soft_van = (
            diff_soft_van.norm() / out_softmax_m.norm().clamp_min(1e-12)
        ).item()
        max_abs_soft_van = diff_soft_van.abs().max().item()

        for S in args.S_values:
            fmm = AlphaFRYSphericalFMM(
                args.l, kappa=kappa, a=a, num_directions=S, sphere=args.sphere
            ).to(
                device=device, dtype=dtype
            )
            t_fmm = _time_forward(fmm, inputs, warmup=args.warmup, iters=args.iters)
            with torch.no_grad():
                out_approx = fmm(*inputs)

            out_approx_m = out_approx.to(dtype=metric_dtype)

            diff_fmm_van = out_approx_m - out_vanilla_m
            rel_l2_fmm_van = (diff_fmm_van.norm() / out_vanilla_m.norm().clamp_min(1e-12)).item()
            max_abs_fmm_van = diff_fmm_van.abs().max().item()

            diff_fmm_soft = out_approx_m - out_softmax_m
            rel_l2_fmm_soft = (
                diff_fmm_soft.norm() / out_softmax_m.norm().clamp_min(1e-12)
            ).item()
            max_abs_fmm_soft = diff_fmm_soft.abs().max().item()

            speedup_vanilla = t_vanilla / max(t_fmm, 1e-12)
            rows.append(
                Row(
                    N=N,
                    S=S,
                    t_softmax=t_softmax,
                    t_vanilla=t_vanilla,
                    t_fmm=t_fmm,
                    speedup_vanilla=speedup_vanilla,
                    rel_l2_soft_van=rel_l2_soft_van,
                    rel_l2_fmm_van=rel_l2_fmm_van,
                    rel_l2_fmm_soft=rel_l2_fmm_soft,
                    max_abs_soft_van=max_abs_soft_van,
                    max_abs_fmm_van=max_abs_fmm_van,
                    max_abs_fmm_soft=max_abs_fmm_soft,
                )
            )

    # Print a compact table
    print("== Sweep benchmark: alpha * f(r) * Y ==")
    print(
        f"device={device} dtype={dtype} l={args.l} Q={args.Q} heads={args.heads} d={args.d} C={args.C} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        "N,S,t_softmax_ms,t_vanilla_ms,t_fmm_ms,speedup_vanilla,"
        "rel_l2_soft_van,rel_l2_fmm_van,rel_l2_fmm_soft"
    )
    for r in rows:
        print(
            f"{r.N},{r.S},{r.t_softmax*1e3:.3f},{r.t_vanilla*1e3:.3f},{r.t_fmm*1e3:.3f},"
            f"{r.speedup_vanilla:.3f},"
            f"{r.rel_l2_soft_van:.4e},{r.rel_l2_fmm_van:.4e},{r.rel_l2_fmm_soft:.4e}"
        )

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(
                "N,S,t_softmax_s,t_vanilla_s,t_fmm_s,speedup_vanilla,"
                "rel_l2_soft_van,rel_l2_fmm_van,rel_l2_fmm_soft,"
                "max_abs_soft_van,max_abs_fmm_van,max_abs_fmm_soft\n"
            )
            for r in rows:
                f.write(
                    f"{r.N},{r.S},{r.t_softmax:.8e},{r.t_vanilla:.8e},{r.t_fmm:.8e},"
                    f"{r.speedup_vanilla:.6f},"
                    f"{r.rel_l2_soft_van:.8e},{r.rel_l2_fmm_van:.8e},{r.rel_l2_fmm_soft:.8e},"
                    f"{r.max_abs_soft_van:.8e},{r.max_abs_fmm_van:.8e},{r.max_abs_fmm_soft:.8e}\n"
                )

    # Plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_speed, ax_err) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    any_positive_err = any(
        math.isfinite(x) and x > 0.0 for r in rows for x in (r.rel_l2_fmm_van, r.rel_l2_soft_van)
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for idx, N in enumerate(args.Ns):
        sub = [r for r in rows if r.N == N]
        sub = sorted(sub, key=lambda rr: rr.S)
        Svals = [r.S for r in sub]
        color = colors[idx % len(colors)]

        speed_van = [r.speedup_vanilla for r in sub]
        ax_speed.plot(Svals, speed_van, marker="o", color=color, label=f"N={N} vanilla/fmm")

        err_fmm_van = [r.rel_l2_fmm_van for r in sub]
        err_soft_van = [r.rel_l2_soft_van for r in sub]
        ax_err.plot(Svals, err_fmm_van, marker="o", color=color, label=f"N={N} fmm-vanilla")
        ax_err.plot(
            Svals, err_soft_van, marker="x", linestyle="--", color=color, label=f"N={N} softmax-vanilla"
        )

    ax_speed.set_xscale("log", base=2)
    ax_speed.set_xlabel("S (# sphere directions)")
    ax_speed.set_ylabel("Speedup (baseline / FMM)")
    ax_speed.grid(True, which="both", alpha=0.3)
    ax_speed.legend()

    ax_err.set_xscale("log", base=2)
    if any_positive_err:
        ax_err.set_yscale("log")
    ax_err.set_xlabel("S (# sphere directions)")
    ax_err.set_ylabel("Relative L2 error")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend()

    fig.suptitle("FMM prototype sweep: softmax vs vanilla(linear) vs FMM")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
