# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from e3nn import o3


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep FMM kappa settings on MD22 geometries and rank by rotation-equivariance error."
        )
    )
    parser.add_argument(
        "--npz-glob",
        type=str,
        default="outputs/md22_full_lmdb/npz/*.npz",
        help="Glob for MD22 npz files.",
    )
    parser.add_argument(
        "--samples-per-mol",
        type=int,
        default=2,
        help="Number of geometry frames sampled per molecule file.",
    )
    parser.add_argument(
        "--rotation-trials",
        type=int,
        default=2,
        help="Number of random rotations per sampled frame.",
    )
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--head-dim", type=int, default=32, help="Per-head q/k dim.")
    parser.add_argument("--value-dim", type=int, default=32, help="Per-head value dim.")
    parser.add_argument(
        "--l-values",
        type=str,
        default="0,1,2",
        help="Comma-separated l values for multi-l FMM core.",
    )
    parser.add_argument(
        "--num-kappa-values",
        type=str,
        default="6,8,10",
        help="Comma-separated candidate num_kappa values.",
    )
    parser.add_argument(
        "--kappa-min-values",
        type=str,
        default="0.8,1.0",
        help="Comma-separated candidate kappa_min values.",
    )
    parser.add_argument(
        "--kappa-max-values",
        type=str,
        default="1.2,1.4,1.8",
        help="Comma-separated candidate kappa_max values.",
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=25,
        help="Spherical quadrature directions (match model default: 25).",
    )
    parser.add_argument(
        "--sphere",
        type=str,
        default="gauss_legendre",
        choices=["gauss_legendre", "lebedev", "fibonacci", "e3nn_s2grid"],
    )
    parser.add_argument("--seed", type=int, default=48, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="outputs/md22_kappa_sweep_equivariance.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


@dataclass
class SampleRecord:
    tag: str
    pos: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    rotations: list[torch.Tensor]


@dataclass
class SweepRow:
    num_kappa: int
    kappa_min: float
    kappa_max: float
    mean_rel_err: float
    p90_rel_err: float
    max_rel_err: float
    elapsed_s: float


def _rand_rotation(generator: torch.Generator) -> torch.Tensor:
    mat = torch.randn(3, 3, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(mat)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _discover_npz_paths(pattern: str) -> list[Path]:
    paths = sorted(Path().glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No NPZ files matched pattern: {pattern}")
    return paths


def _build_samples(
    *,
    npz_paths: list[Path],
    samples_per_mol: int,
    rotation_trials: int,
    heads: int,
    head_dim: int,
    value_dim: int,
    seed: int,
) -> list[SampleRecord]:
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed + 13)
    records: list[SampleRecord] = []

    for npz_path in npz_paths:
        with np.load(npz_path) as arr:
            positions = arr["R"].astype(np.float32)  # (n_frames, n_atoms, 3)
        n_frames = int(positions.shape[0])
        n_pick = min(samples_per_mol, n_frames)
        pick_idx = rng.choice(n_frames, size=n_pick, replace=False)
        pick_idx.sort()
        for frame_idx in pick_idx.tolist():
            pos_np = positions[frame_idx]
            n_atoms = int(pos_np.shape[0])
            q = torch.randn(n_atoms, heads, head_dim, generator=torch_gen, dtype=torch.float32)
            k = torch.randn(n_atoms, heads, head_dim, generator=torch_gen, dtype=torch.float32)
            v = torch.randn(n_atoms, heads, value_dim, generator=torch_gen, dtype=torch.float32)
            rots = [_rand_rotation(torch_gen) for _ in range(rotation_trials)]
            records.append(
                SampleRecord(
                    tag=f"{npz_path.name}:frame{frame_idx}",
                    pos=torch.from_numpy(pos_np),
                    q=q,
                    k=k,
                    v=v,
                    rotations=rots,
                )
            )
    if len(records) == 0:
        raise RuntimeError("No samples collected for sweep.")
    return records


@torch.no_grad()
def _evaluate_candidate(
    *,
    samples: list[SampleRecord],
    l_values: list[int],
    num_kappa: int,
    kappa_min: float,
    kappa_max: float,
    num_directions: int,
    sphere: str,
    device: torch.device,
) -> SweepRow:
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMMMultiL  # noqa: WPS433

    kappa = torch.linspace(kappa_min, kappa_max, num_kappa, dtype=torch.float32)
    a = torch.ones(samples[0].q.shape[1], num_kappa, dtype=torch.float32) / math.sqrt(float(num_kappa))

    core = AlphaFRYSphericalFMMMultiL(
        l_values=l_values,
        kappa=kappa,
        a=a,
        num_directions=num_directions,
        sphere=sphere,
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
        optimize_low_precision_sphere=False,
        center_positions=True,
    ).to(device)
    core.eval()

    rel_errors: list[float] = []
    t0 = time.perf_counter()
    for record in samples:
        pos = record.pos.to(device)
        q = record.q.to(device)
        k = record.k.to(device)
        v = record.v.to(device)
        node_mask = torch.ones(1, pos.shape[0], dtype=torch.bool, device=device)

        out_base = core(
            pos.unsqueeze(0),
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            node_mask=node_mask,
        )
        for rot in record.rotations:
            rot_dev = rot.to(device=device, dtype=pos.dtype)
            pos_rot = torch.matmul(pos, rot_dev.t())
            out_rot = core(
                pos_rot.unsqueeze(0),
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                node_mask=node_mask,
            )
            for l_idx, l_val in enumerate(l_values):
                base_l = out_base[l_idx][0]  # (n, h, m, c)
                rot_l = out_rot[l_idx][0]    # (n, h, m, c)
                d_mat = o3.Irrep(l_val, 1).D_from_matrix(rot.cpu()).to(
                    device=device,
                    dtype=base_l.dtype,
                )
                expected = torch.einsum("pm,nhmc->nhpc", d_mat, base_l)
                expected_t = torch.einsum("mp,nhmc->nhpc", d_mat, base_l)
                err = (rot_l - expected).norm() / expected.norm().clamp_min(1e-12)
                err_t = (rot_l - expected_t).norm() / expected_t.norm().clamp_min(1e-12)
                rel_errors.append(float(torch.minimum(err, err_t).item()))
    elapsed = time.perf_counter() - t0
    err_tensor = torch.tensor(rel_errors, dtype=torch.float64)
    mean_err = float(err_tensor.mean().item())
    p90_err = float(torch.quantile(err_tensor, 0.9).item())
    max_err = float(err_tensor.max().item())
    return SweepRow(
        num_kappa=num_kappa,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        mean_rel_err=mean_err,
        p90_rel_err=p90_err,
        max_rel_err=max_err,
        elapsed_s=elapsed,
    )


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    l_values = _parse_int_list(args.l_values)
    if len(l_values) == 0:
        raise ValueError("--l-values must be non-empty.")
    num_kappa_values = _parse_int_list(args.num_kappa_values)
    kappa_min_values = _parse_float_list(args.kappa_min_values)
    kappa_max_values = _parse_float_list(args.kappa_max_values)

    npz_paths = _discover_npz_paths(args.npz_glob)
    samples = _build_samples(
        npz_paths=npz_paths,
        samples_per_mol=int(args.samples_per_mol),
        rotation_trials=int(args.rotation_trials),
        heads=int(args.heads),
        head_dim=int(args.head_dim),
        value_dim=int(args.value_dim),
        seed=int(args.seed),
    )

    candidates: list[tuple[int, float, float]] = []
    for num_kappa, kappa_min, kappa_max in itertools.product(
        num_kappa_values,
        kappa_min_values,
        kappa_max_values,
    ):
        if num_kappa <= 0:
            continue
        if kappa_max <= kappa_min:
            continue
        candidates.append((int(num_kappa), float(kappa_min), float(kappa_max)))
    if len(candidates) == 0:
        raise ValueError("No valid kappa candidates after filtering.")

    print("== MD22 kappa equivariance sweep ==")
    print(
        f"device={device} npz_files={len(npz_paths)} samples={len(samples)} "
        f"rot_trials={args.rotation_trials} l_values={l_values} "
        f"heads={args.heads} d={args.head_dim} c={args.value_dim} "
        f"dirs={args.num_directions} sphere={args.sphere}"
    )
    print(f"candidates={len(candidates)}")

    rows: list[SweepRow] = []
    for idx, (num_kappa, kappa_min, kappa_max) in enumerate(candidates, start=1):
        print(
            f"[{idx:02d}/{len(candidates):02d}] "
            f"num_kappa={num_kappa} kappa_min={kappa_min:.3f} kappa_max={kappa_max:.3f}"
        )
        row = _evaluate_candidate(
            samples=samples,
            l_values=l_values,
            num_kappa=num_kappa,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            num_directions=int(args.num_directions),
            sphere=args.sphere,
            device=device,
        )
        rows.append(row)
        print(
            f"    mean_rel_err={row.mean_rel_err:.6e} "
            f"p90={row.p90_rel_err:.6e} max={row.max_rel_err:.6e} "
            f"time={row.elapsed_s:.2f}s"
        )

    rows.sort(key=lambda row: (row.mean_rel_err, row.p90_rel_err, row.max_rel_err))
    best = rows[0]
    print()
    print("Top candidates by mean relative equivariance error:")
    for rank, row in enumerate(rows[:5], start=1):
        print(
            f"#{rank}: num_kappa={row.num_kappa} "
            f"kappa_min={row.kappa_min:.3f} kappa_max={row.kappa_max:.3f} "
            f"mean={row.mean_rel_err:.6e} p90={row.p90_rel_err:.6e} "
            f"max={row.max_rel_err:.6e}"
        )
    print()
    print(
        f"BEST: num_kappa={best.num_kappa} "
        f"kappa_min={best.kappa_min:.3f} kappa_max={best.kappa_max:.3f} "
        f"mean_rel_err={best.mean_rel_err:.6e}"
    )

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "num_kappa",
                "kappa_min",
                "kappa_max",
                "mean_rel_err",
                "p90_rel_err",
                "max_rel_err",
                "elapsed_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.num_kappa,
                    row.kappa_min,
                    row.kappa_max,
                    row.mean_rel_err,
                    row.p90_rel_err,
                    row.max_rel_err,
                    row.elapsed_s,
                ]
            )
    print(f"Saved sweep results to {csv_path}")


if __name__ == "__main__":
    main()
