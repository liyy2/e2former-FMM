# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
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
            "Functional checks for AlphaFRYSphericalFMM: permutation/translation "
            "invariance, chunk/phase consistency, headed-unheaded consistency, "
            "and agreement with exact linear baseline on small cases."
        )
    )
    parser.add_argument(
        "--max-nodes",
        "--num-nodes",
        dest="max_nodes",
        type=int,
        default=24,
        help="Padded nodes per graph.",
    )
    parser.add_argument("--heads", type=int, default=4, help="Number of heads.")
    parser.add_argument("--head-dim", type=int, default=16, help="Per-head q/k dim.")
    parser.add_argument("--value-dim", type=int, default=8, help="Per-head value dim.")
    parser.add_argument("--degree", type=int, default=1, help="Spherical harmonic degree l.")
    parser.add_argument("--kappa", type=int, default=8, help="Number of radial frequencies.")
    parser.add_argument("--dirs", type=int, default=49, help="Directions for regular FMM tests.")
    parser.add_argument(
        "--dirs-exact-check",
        type=int,
        default=121,
        help="Directions used for FMM-vs-exact test (larger => better approximation).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tol-strict", type=float, default=5e-5)
    parser.add_argument("--tol-chunk", type=float, default=2e-4)
    parser.add_argument("--tol-phase", type=float, default=2e-4)
    parser.add_argument(
        "--tol-rot-fmm",
        type=float,
        default=2e-1,
        help="Approximate equivariance tolerance for FMM (quadrature-limited).",
    )
    parser.add_argument("--tol-rot-softmax", type=float, default=1e-5)
    parser.add_argument(
        "--check-rotation-sweep",
        action="store_true",
        help="Verify that FMM rotation error decreases with larger num_directions.",
    )
    parser.add_argument(
        "--rot-sweep-dirs",
        type=str,
        default="25,49,81,121",
        help="Comma-separated num_directions values for rotation sweep.",
    )
    parser.add_argument(
        "--rot-sweep-trials",
        type=int,
        default=4,
        help="Random rotation trials per num_directions in sweep.",
    )
    parser.add_argument(
        "--rot-sweep-min-drop",
        type=float,
        default=0.1,
        help="Required relative error drop from first to last sweep point.",
    )
    parser.add_argument(
        "--rot-sweep-abs-tol",
        type=float,
        default=1e-6,
        help="Absolute slack when comparing final sweep error to best earlier error.",
    )
    parser.add_argument("--max-rel-fmm-exact", type=float, default=0.25)
    parser.add_argument("--min-cos-fmm-exact", type=float, default=0.90)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def _expand_node_mask(node_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    m = node_mask
    for _ in range(ref.ndim - node_mask.ndim):
        m = m.unsqueeze(-1)
    return m.expand_as(ref)


def _masked_max_abs(x: torch.Tensor, node_mask: torch.Tensor) -> float:
    m = _expand_node_mask(node_mask, x)
    if bool(m.any()):
        return float(x[m].abs().max().item())
    return 0.0


def _permute_nodes(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    b_idx = torch.arange(x.shape[0], device=x.device)[:, None]
    return x[b_idx, perm]


def _assert_leq(name: str, value: float, bound: float) -> None:
    if value > bound:
        raise RuntimeError(f"{name}={value:.6e} exceeds bound={bound:.6e}")


def _parse_dirs_list(text: str) -> list[int]:
    items = [part.strip() for part in text.split(",") if part.strip()]
    dirs = [int(item) for item in items]
    if len(dirs) < 2:
        raise ValueError(
            f"Need at least two sweep values in --rot-sweep-dirs, got: {text!r}"
        )
    if any(val <= 0 for val in dirs):
        raise ValueError(f"All sweep directions must be positive, got: {dirs}")
    if any(dirs[i] >= dirs[i + 1] for i in range(len(dirs) - 1)):
        raise ValueError(
            f"--rot-sweep-dirs must be strictly increasing, got: {dirs}"
        )
    return dirs


@torch.no_grad()
def _rotation_errors_for_core(
    *,
    core: torch.nn.Module,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    l: int,
    trials: int,
) -> list[float]:
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")

    out = core(pos, q, k, v, node_mask=node_mask)  # (B,N_nodes,H,M,C)
    irrep = o3.Irrep(l, 1)
    errors: list[float] = []
    for _ in range(trials):
        rot = o3.rand_matrix().to(device=pos.device, dtype=pos.dtype)  # (3,3)
        pos_rot = torch.matmul(pos, rot.t())
        out_rot = core(pos_rot, q, k, v, node_mask=node_mask)

        d_mat = irrep.D_from_matrix(rot.cpu()).to(device=out.device, dtype=out.dtype)
        out_expected = torch.einsum("mn,blhnc->blhmc", d_mat, out)
        out_expected_t = torch.einsum("nm,blhnc->blhmc", d_mat, out)
        err_direct = _masked_max_abs(out_rot - out_expected, node_mask=node_mask)
        err_transpose = _masked_max_abs(out_rot - out_expected_t, node_mask=node_mask)
        errors.append(min(err_direct, err_transpose))
    return errors


@torch.no_grad()
def _test_permutation_equivariance(
    core: torch.nn.Module,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    tol: float,
) -> float:
    out = core(pos, q, k, v, node_mask=node_mask)
    bsz, n_nodes = node_mask.shape
    perm = torch.stack(
        [torch.randperm(n_nodes, device=pos.device) for _ in range(bsz)], dim=0
    )
    inv_perm = torch.argsort(perm, dim=1)

    out_perm = core(
        _permute_nodes(pos, perm),
        _permute_nodes(q, perm),
        _permute_nodes(k, perm),
        _permute_nodes(v, perm),
        node_mask=_permute_nodes(node_mask, perm),
    )
    out_recovered = _permute_nodes(out_perm, inv_perm)
    err = _masked_max_abs(out - out_recovered, node_mask=node_mask)
    _assert_leq("permutation_err", err, tol)
    return err


@torch.no_grad()
def _test_translation_invariance(
    core: torch.nn.Module,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    tol: float,
) -> float:
    out = core(pos, q, k, v, node_mask=node_mask)
    shift = torch.randn(pos.shape[0], 1, 3, device=pos.device, dtype=pos.dtype)
    out_shift = core(pos + shift, q, k, v, node_mask=node_mask)
    err = _masked_max_abs(out - out_shift, node_mask=node_mask)
    _assert_leq("translation_err", err, tol)
    return err


@torch.no_grad()
def _test_headed_unheaded_equivalence(
    *,
    l: int,
    kappa: torch.Tensor,
    dirs: int,
    pos: torch.Tensor,
    node_mask: torch.Tensor,
    head_dim: int,
    value_dim: int,
    tol: float,
) -> float:
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    q_u = torch.randn(pos.shape[0], pos.shape[1], head_dim, device=pos.device)
    k_u = torch.randn(pos.shape[0], pos.shape[1], head_dim, device=pos.device)
    v_u = torch.randn(pos.shape[0], pos.shape[1], value_dim, device=pos.device)

    a = torch.randn(1, int(kappa.shape[0]), dtype=torch.float32)
    core = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(pos.device)
    core.eval()

    out_u = core(pos, q_u, k_u, v_u, node_mask=node_mask)  # (B,N_nodes,M,C)
    out_h = core(
        pos,
        q_u.unsqueeze(2),
        k_u.unsqueeze(2),
        v_u.unsqueeze(2),
        node_mask=node_mask,
    ).squeeze(2)  # (B,N_nodes,M,C)
    err = _masked_max_abs(out_u - out_h, node_mask=node_mask)
    _assert_leq("headed_unheaded_err", err, tol)
    return err


@torch.no_grad()
def _test_chunk_consistency(
    *,
    l: int,
    kappa: torch.Tensor,
    a: torch.Tensor,
    dirs: int,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    tol: float,
) -> float:
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    core_chunk1 = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=1,
        promote_half_precision=False,
    ).to(pos.device)
    core_chunkq = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=int(kappa.shape[0]),
        promote_half_precision=False,
    ).to(pos.device)
    core_chunk1.eval()
    core_chunkq.eval()

    out1 = core_chunk1(pos, q, k, v, node_mask=node_mask)
    outq = core_chunkq(pos, q, k, v, node_mask=node_mask)
    err = _masked_max_abs(out1 - outq, node_mask=node_mask)
    _assert_leq("chunk_consistency_err", err, tol)
    return err


@torch.no_grad()
def _test_phase_mode_consistency(
    *,
    l: int,
    kappa: torch.Tensor,
    a: torch.Tensor,
    dirs: int,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    tol: float,
) -> float:
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    core_trig = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(pos.device)
    core_complex = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="complex",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(pos.device)
    core_trig.eval()
    core_complex.eval()

    out_trig = core_trig(pos, q, k, v, node_mask=node_mask)
    out_complex = core_complex(pos, q, k, v, node_mask=node_mask)
    err = _masked_max_abs(out_trig - out_complex, node_mask=node_mask)
    _assert_leq("phase_mode_err", err, tol)
    return err


@torch.no_grad()
def _test_rotation_equivariance(
    *,
    core: torch.nn.Module,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    l: int,
    tol: float,
    trials: int = 2,
) -> float:
    errors = _rotation_errors_for_core(
        core=core,
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        l=l,
        trials=trials,
    )
    worst_err = max(errors)
    _assert_leq("rotation_equivariance_err", worst_err, tol)
    return worst_err


@torch.no_grad()
def _test_rotation_error_decreases_with_dirs(
    *,
    l: int,
    kappa: torch.Tensor,
    a: torch.Tensor,
    pos: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    node_mask: torch.Tensor,
    dirs_list: list[int],
    trials: int,
    min_drop: float,
    abs_tol: float,
) -> tuple[list[int], list[float]]:
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    means: list[float] = []
    for dirs in dirs_list:
        core = AlphaFRYSphericalFMM(
            l=l,
            kappa=kappa,
            a=a,
            num_directions=int(dirs),
            sphere="gauss_legendre",
            phase_mode="trig",
            kappa_chunk_size=0,
            promote_half_precision=False,
        ).to(pos.device)
        core.eval()
        errs = _rotation_errors_for_core(
            core=core,
            pos=pos,
            q=q,
            k=k,
            v=v,
            node_mask=node_mask,
            l=l,
            trials=trials,
        )
        means.append(float(sum(errs) / len(errs)))

    first = means[0]
    last = means[-1]
    required_last = first * (1.0 - min_drop)
    if last > required_last:
        raise RuntimeError(
            f"rotation_sweep final mean error {last:.6e} does not improve enough "
            f"from first {first:.6e}; required <= {required_last:.6e}."
        )
    best_earlier = min(means[:-1])
    if last > best_earlier + abs_tol:
        raise RuntimeError(
            f"rotation_sweep final mean error {last:.6e} is not the best within slack; "
            f"best earlier {best_earlier:.6e}, abs_tol={abs_tol:.6e}."
        )
    return dirs_list, means


@torch.no_grad()
def _test_fmm_vs_exact_linear(
    *,
    kappa: torch.Tensor,
    dirs: int,
    head_dim: int,
    value_dim: int,
    max_rel: float,
    min_cos: float,
    device: torch.device,
) -> tuple[float, float]:
    from molfm.models.e2former.fmm_exact import AlphaFRYSphericalLinear  # noqa: WPS433
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433

    bsz, n_nodes, num_heads, l = 1, 12, 2, 0
    pos = torch.randn(bsz, n_nodes, 3, device=device)
    q = torch.randn(bsz, n_nodes, num_heads, head_dim, device=device)
    k = torch.randn(bsz, n_nodes, num_heads, head_dim, device=device)
    v = torch.randn(bsz, n_nodes, num_heads, value_dim, device=device)
    node_mask = torch.ones(bsz, n_nodes, dtype=torch.bool, device=device)
    a = torch.randn(num_heads, int(kappa.shape[0]), dtype=torch.float32)

    fmm = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=dirs,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(device)
    exact = AlphaFRYSphericalLinear(
        l=l,
        kappa=kappa,
        a=a,
        feature_map="elu",
    ).to(device)
    fmm.eval()
    exact.eval()

    out_fmm = fmm(pos, q, k, v, node_mask=node_mask)
    out_exact = exact(pos, q, k, v, node_mask=node_mask)

    diff = out_fmm - out_exact
    rel = float(diff.norm() / out_exact.norm().clamp_min(1e-12))
    cos = float(
        F.cosine_similarity(
            out_fmm.reshape(1, -1), out_exact.reshape(1, -1), dim=1
        ).item()
    )
    if rel > max_rel:
        raise RuntimeError(f"fmm_vs_exact_rel={rel:.6e} exceeds max_rel={max_rel:.6e}")
    if cos < min_cos:
        raise RuntimeError(f"fmm_vs_exact_cos={cos:.6e} below min_cos={min_cos:.6e}")
    return rel, cos


@torch.no_grad()
def _test_wrapper_shapes(
    *,
    kappa: torch.Tensor,
    device: torch.device,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    from molfm.models.e2former.fmm_prototype import E2FormerAlphaFRYPrototype  # noqa: WPS433

    bsz, n_nodes, lmax, in_channels = 2, 10, 2, 6
    l = 1
    num_heads = 2
    value_dim = 3
    node_pos = torch.randn(bsz, n_nodes, 3, device=device)
    node_irreps = torch.randn(bsz, n_nodes, (lmax + 1) ** 2, in_channels, device=device)
    node_mask = torch.ones(bsz, n_nodes, dtype=torch.bool, device=device)
    a = torch.randn(num_heads, int(kappa.shape[0]), dtype=torch.float32)

    proto_merge = E2FormerAlphaFRYPrototype(
        l=l,
        kappa=kappa,
        a=a,
        method="fmm",
        num_heads=num_heads,
        head_dim=8,
        value_dim=value_dim,
        num_directions=25,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        lmax=lmax,
        in_channels=in_channels,
        merge_heads=True,
        return_embedding=False,
    ).to(device)
    proto_keep = E2FormerAlphaFRYPrototype(
        l=l,
        kappa=kappa,
        a=a,
        method="fmm",
        num_heads=num_heads,
        head_dim=8,
        value_dim=value_dim,
        num_directions=25,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        lmax=lmax,
        in_channels=in_channels,
        merge_heads=False,
        return_embedding=False,
    ).to(device)
    proto_emb = E2FormerAlphaFRYPrototype(
        l=l,
        kappa=kappa,
        a=a,
        method="fmm",
        num_heads=num_heads,
        head_dim=8,
        value_dim=value_dim,
        num_directions=25,
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        lmax=lmax,
        in_channels=in_channels,
        merge_heads=True,
        return_embedding=True,
    ).to(device)
    proto_merge.eval()
    proto_keep.eval()
    proto_emb.eval()

    out_merge = proto_merge(node_pos, node_irreps, node_mask=node_mask)
    out_keep = proto_keep(node_pos, node_irreps, node_mask=node_mask)
    out_emb = proto_emb(node_pos, node_irreps, node_mask=node_mask)

    expected_merge = (bsz, n_nodes, 2 * l + 1, num_heads * value_dim)
    expected_keep = (bsz, n_nodes, 2 * l + 1, num_heads, value_dim)
    expected_emb = (bsz, n_nodes, (lmax + 1) ** 2, num_heads * value_dim)
    if tuple(out_merge.shape) != expected_merge:
        raise RuntimeError(f"wrapper_merge_shape got {tuple(out_merge.shape)}, expected {expected_merge}")
    if tuple(out_keep.shape) != expected_keep:
        raise RuntimeError(f"wrapper_keep_shape got {tuple(out_keep.shape)}, expected {expected_keep}")
    if tuple(out_emb.shape) != expected_emb:
        raise RuntimeError(f"wrapper_emb_shape got {tuple(out_emb.shape)}, expected {expected_emb}")

    start = l * l
    end = (l + 1) * (l + 1)
    if start > 0:
        if float(out_emb[:, :, :start, :].abs().max().item()) > 1e-6:
            raise RuntimeError("wrapper embedding has non-zero entries below target l-slice")
    if end < out_emb.shape[2]:
        if float(out_emb[:, :, end:, :].abs().max().item()) > 1e-6:
            raise RuntimeError("wrapper embedding has non-zero entries above target l-slice")
    return tuple(out_merge.shape), tuple(out_keep.shape), tuple(out_emb.shape)


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    bsz = 2
    n_nodes = int(args.max_nodes)
    num_heads = int(args.heads)
    head_dim = int(args.head_dim)
    value_dim = int(args.value_dim)
    l = int(args.degree)
    q_num = int(args.kappa)

    pos = torch.randn(bsz, n_nodes, 3, device=device)
    q = torch.randn(bsz, n_nodes, num_heads, head_dim, device=device)
    k = torch.randn(bsz, n_nodes, num_heads, head_dim, device=device)
    v = torch.randn(bsz, n_nodes, num_heads, value_dim, device=device)
    node_mask = torch.zeros(bsz, n_nodes, dtype=torch.bool, device=device)
    node_mask[0, : max(2, n_nodes - 3)] = True
    node_mask[1, : max(2, n_nodes // 3)] = True

    kappa = torch.linspace(1.0, float(q_num), q_num, dtype=torch.float32)
    a = torch.randn(num_heads, q_num, dtype=torch.float32)

    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalFMM  # noqa: WPS433
    from molfm.models.e2former.fmm_prototype import AlphaFRYSphericalSoftmax  # noqa: WPS433

    core = AlphaFRYSphericalFMM(
        l=l,
        kappa=kappa,
        a=a,
        num_directions=int(args.dirs),
        sphere="gauss_legendre",
        phase_mode="trig",
        kappa_chunk_size=0,
        promote_half_precision=False,
    ).to(device)
    softmax_core = AlphaFRYSphericalSoftmax(
        l=l,
        kappa=kappa,
        a=a,
    ).to(device)
    core.eval()
    softmax_core.eval()

    print("== FMM functionality checks ==")
    print(
        f"device={device} B={bsz} num_nodes={n_nodes} valid_lengths="
        f"[{int(node_mask[0].sum().item())}, {int(node_mask[1].sum().item())}] "
        f"heads={num_heads} head_dim={head_dim} value_dim={value_dim} l={l}"
    )

    perm_err = _test_permutation_equivariance(
        core, pos, q, k, v, node_mask, tol=float(args.tol_strict)
    )
    trans_err = _test_translation_invariance(
        core, pos, q, k, v, node_mask, tol=float(args.tol_strict)
    )
    headed_err = _test_headed_unheaded_equivalence(
        l=l,
        kappa=kappa,
        dirs=int(args.dirs),
        pos=pos,
        node_mask=node_mask,
        head_dim=head_dim,
        value_dim=value_dim,
        tol=float(args.tol_strict),
    )
    chunk_err = _test_chunk_consistency(
        l=l,
        kappa=kappa,
        a=a,
        dirs=int(args.dirs),
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        tol=float(args.tol_chunk),
    )
    phase_err = _test_phase_mode_consistency(
        l=l,
        kappa=kappa,
        a=a,
        dirs=int(args.dirs),
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        tol=float(args.tol_phase),
    )
    rot_fmm_err = _test_rotation_equivariance(
        core=core,
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        l=l,
        tol=float(args.tol_rot_fmm),
    )
    rot_softmax_err = _test_rotation_equivariance(
        core=softmax_core,
        pos=pos,
        q=q,
        k=k,
        v=v,
        node_mask=node_mask,
        l=l,
        tol=float(args.tol_rot_softmax),
    )
    rot_sweep_dirs: list[int] = []
    rot_sweep_means: list[float] = []
    if args.check_rotation_sweep:
        dirs_list = _parse_dirs_list(args.rot_sweep_dirs)
        rot_sweep_dirs, rot_sweep_means = _test_rotation_error_decreases_with_dirs(
            l=l,
            kappa=kappa,
            a=a,
            pos=pos,
            q=q,
            k=k,
            v=v,
            node_mask=node_mask,
            dirs_list=dirs_list,
            trials=int(args.rot_sweep_trials),
            min_drop=float(args.rot_sweep_min_drop),
            abs_tol=float(args.rot_sweep_abs_tol),
        )
    rel_exact, cos_exact = _test_fmm_vs_exact_linear(
        kappa=kappa,
        dirs=int(args.dirs_exact_check),
        head_dim=head_dim,
        value_dim=value_dim,
        max_rel=float(args.max_rel_fmm_exact),
        min_cos=float(args.min_cos_fmm_exact),
        device=device,
    )
    shape_merge, shape_keep, shape_emb = _test_wrapper_shapes(kappa=kappa, device=device)

    print(f"permutation_err:         {perm_err:.6e}")
    print(f"translation_err:         {trans_err:.6e}")
    print(f"headed_unheaded_err:     {headed_err:.6e}")
    print(f"chunk_consistency_err:   {chunk_err:.6e}")
    print(f"phase_mode_err:          {phase_err:.6e}")
    print(f"rotation_fmm_err:        {rot_fmm_err:.6e}")
    print(f"rotation_softmax_err:    {rot_softmax_err:.6e}")
    if args.check_rotation_sweep:
        print("rotation_sweep_mean_err:")
        for dirs, mean_err in zip(rot_sweep_dirs, rot_sweep_means):
            print(f"  dirs={dirs:4d} -> {mean_err:.6e}")
    print(f"fmm_vs_exact_rel:        {rel_exact:.6e}")
    print(f"fmm_vs_exact_cos:        {cos_exact:.6e}")
    print(f"wrapper_merge_shape:     {shape_merge}")
    print(f"wrapper_keep_shape:      {shape_keep}")
    print(f"wrapper_embedding_shape: {shape_emb}")


if __name__ == "__main__":
    main()
