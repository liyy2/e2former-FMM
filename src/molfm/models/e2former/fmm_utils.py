"""Utility functions shared by the FMM and exact attention operators.

This module provides:
    positive_feature_map:  Non-negative feature maps for linear attention kernels.
    prepare_radial_coeffs: Validates and normalizes radial mixing coefficient shapes.
    spherical_bessel_j:    Differentiable spherical Bessel functions j_l(x) via recurrence.
    prod:                  Integer product for flattening tensor leading dimensions.
    infer_compute_dtype:   Chooses fp32 under AMP for geometry-heavy math.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from functools import lru_cache
from typing import Literal

import torch


def positive_feature_map(x: torch.Tensor, kind: str = "elu") -> torch.Tensor:
    """Map arbitrary real-valued features to the non-negative half-line.

    This is required for **linear attention**, which replaces the softmax with a
    kernel factorization:

        softmax(qk/d)    <(q), (k)> / Z

    For the denominator Z = _j <(q_i), (k_j)> to remain well-behaved (positive
    and bounded away from zero), the feature map  must produce non-negative outputs.

    Available maps:
        "elu":      (x) = ELU(x) + 1.  Smooth, strictly positive.  Gradients are
                    non-zero everywhere.  This is the default used by Katharopoulos
                    et al. ("Transformers are RNNs", 2020).
        "softplus": (x) = log(1 + exp(x)).  Always positive, smoother tail behavior
                    for large negative inputs.

    Args:
        x:    Input tensor of any shape.
        kind: Feature map name ("elu" or "softplus").

    Returns:
        Non-negative tensor of the same shape as x.
    """
    if kind == "elu":
        return torch.nn.functional.elu(x) + 1.0
    if kind == "softplus":
        return torch.nn.functional.softplus(x)
    raise ValueError(f"Unknown feature_map kind: {kind!r}")


def prepare_radial_coeffs(
    a: torch.Tensor,
    *,
    num_kappa: int,
    num_heads: int,
    value_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Literal["scalar", "head", "head_channel"]]:
    """Validate and place radial mixing coefficients on the correct device/dtype.

    The radial function f_l(r) = _q a_q  j_l(_q  r) can have coefficients
    that are shared globally, per-head, or per-head-per-channel:

        - scalar (a.ndim=1):       a.shape = (Q,)
          All heads and value channels share the same radial profile.

        - per-head (a.ndim=2):     a.shape = (H, Q)
          Each attention head has its own radial profile, but all value channels
          within a head share it. This allows heads to specialize in different
          distance scales.

        - per-head-channel (a.ndim=3): a.shape = (H, C, Q) where C in {1, value_dim}
          The most expressive mode: each (head, channel) pair has its own profile.
          When C=1, the single set of coefficients is broadcast over all channels.

    Args:
        a:          Radial mixing coefficients tensor.
        num_kappa:  Expected number of radial frequencies Q (must match a.shape[-1]).
        num_heads:  Current number of attention heads H.
        value_dim:  Current value dimension Cv (for validating C in head_channel mode).
        device:     Target device.
        dtype:      Target dtype.

    Returns:
        Tuple of (a_placed, mode_string) where a_placed is on the correct device/dtype
        and mode_string is one of "scalar", "head", or "head_channel".
    """
    if a.ndim == 1:
        # Scalar mode: f_l(r) is the same for all heads and channels.
        if a.shape[0] != num_kappa:
            raise ValueError(
                f"Expected scalar radial coeffs a.shape=(Q,) with Q={num_kappa}, got {tuple(a.shape)}"
            )
        return a.to(device=device, dtype=dtype), "scalar"

    if a.ndim == 2:
        # Per-head mode: each head h gets its own radial profile.
        if a.shape[1] != num_kappa:
            raise ValueError(
                f"Expected per-head radial coeffs a.shape=(H,Q) with Q={num_kappa}, got {tuple(a.shape)}"
            )
        if a.shape[0] != num_heads:
            raise ValueError(
                f"Per-head radial coeffs must match current H={num_heads}, got a.shape={tuple(a.shape)}"
            )
        return a.to(device=device, dtype=dtype), "head"

    if a.ndim == 3:
        # Per-head-per-channel mode: most expressive, each (h,c) has its own profile.
        if a.shape[2] != num_kappa:
            raise ValueError(
                f"Expected per-head-channel radial coeffs a.shape=(H,C,Q) with Q={num_kappa}, got {tuple(a.shape)}"
            )
        if a.shape[0] != num_heads:
            raise ValueError(
                f"Per-head-channel radial coeffs must match current H={num_heads}, got a.shape={tuple(a.shape)}"
            )
        if a.shape[1] not in (1, value_dim):
            raise ValueError(
                f"Per-head-channel coeffs must have C in {{1,{value_dim}}}, got C={a.shape[1]}"
            )
        return a.to(device=device, dtype=dtype), "head_channel"

    raise ValueError(
        f"Unsupported a.ndim={a.ndim}. Expected a with shape (Q,), (H,Q), or (H,C,Q)."
    )


def spherical_bessel_j(l: int, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the spherical Bessel function of the first kind, j_l(x).

    The spherical Bessel functions arise naturally in solutions to the Helmholtz
    equation ( + k) = 0 in spherical coordinates.  They form the radial part
    of the plane-wave expansion:

        e^{ikr} = _l  i^l (2l+1) j_l(kr) P_l(cos )

    which is the mathematical foundation for the FMM factorization.

    Implementation notes:
        - Fully differentiable (torch autograd compatible).
        - Runs on any device (CPU/CUDA)  no SciPy dependency.
        - Uses the stable forward recurrence for l >= 2.
        - The small-|x| regime is handled via known limits:
              j_0(0) = 1,  j_l(0) = 0 for l > 0.
          This avoids the 0/0 indeterminate form in sin(x)/x at x=0.

    Closed forms for l=0,1:
        j_0(x) = sin(x) / x
        j_1(x) = sin(x) / x^2 - cos(x) / x

    Forward recurrence for l >= 2:
        j_{l+1}(x) = ((2l+1)/x)  j_l(x)  -  j_{l-1}(x)

    Args:
        l:   Non-negative integer degree.
        x:   Input tensor of any shape (typically contains _q  r_ij values).
        eps: Threshold below which |x| is considered "near zero" for limit handling.

    Returns:
        Tensor of same shape as x containing j_l(x).
    """
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")

    # Identify elements near the origin where direct formulas would divide by zero.
    x_abs = x.abs()
    small = x_abs < eps

    # --- l = 0: j_0(x) = sin(x)/x ---
    # At x=0, the limit is j_0(0) = 1 (by L'Hopital or Taylor expansion).
    j0 = torch.where(small, torch.ones_like(x), torch.sin(x) / x)
    if l == 0:
        return j0

    # --- l = 1: j_1(x) = sin(x)/x^2 - cos(x)/x ---
    # At x=0, the limit is j_1(0) = 0. We use x_safe to avoid NaN at x=0,
    # then overwrite those entries with the known limit.
    x_safe = torch.where(small, torch.ones_like(x), x)
    j1 = torch.sin(x_safe) / (x_safe * x_safe) - torch.cos(x_safe) / x_safe
    j1 = torch.where(small, torch.zeros_like(x), j1)
    if l == 1:
        return j1

    # --- l >= 2: Forward recurrence ---
    # j_{n+1}(x) = ((2n+1)/x)  j_n(x) - j_{n-1}(x)
    # This recurrence is numerically stable in the forward direction for small l.
    # (For very large l, backward recurrence would be needed, but l < ~10 here.)
    j_minus_1 = j0
    j_0 = j1
    for n in range(1, l):
        j_plus_1 = (2 * n + 1) * j_0 / x_safe - j_minus_1
        # Re-apply the limit j_{l>0}(0) = 0 at near-zero entries.
        j_plus_1 = torch.where(small, torch.zeros_like(x), j_plus_1)
        j_minus_1, j_0 = j_0, j_plus_1
    return j_0


def prod(shape: torch.Size) -> int:
    """Compute the product of all elements in a shape tuple.

    Used throughout the FMM/attention code to flatten arbitrary leading dimensions
    (e.g., B, L) into a single node axis N = B*L, so that the core einsum
    contractions can be written with simple indices (i, j, n) regardless of
    how many batch dimensions the caller uses.

    Example:
        prod(torch.Size([4, 32])) -> 128  (B=4, L=32 -> N=128)
    """
    out = 1
    for s in shape:
        out *= int(s)
    return int(out)


def infer_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Choose a numerically safe compute dtype for geometry-heavy kernels.

    Under Automatic Mixed Precision (AMP), the model's parameters and activations
    may be in fp16 or bf16. However, spherical harmonics, Bessel functions, and
    exp(iphase) computations are sensitive to precision loss. This function
    promotes half-precision dtypes to fp32 while leaving fp64 inputs unchanged.

    Args:
        dtype: The input tensor's dtype (e.g., from pos.dtype).

    Returns:
        torch.float32 if the input is a half-precision type, otherwise the input dtype.
    """
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


@lru_cache(maxsize=1)
def _cueq_ops_available() -> bool:
    """Return True when cuEquivariance torch kernels are importable and usable."""
    if importlib.util.find_spec("cuequivariance_torch") is None:
        return False
    if importlib.util.find_spec("cuequivariance_ops_torch") is None:
        return False
    try:
        # Some broken installs print loader errors to stderr; suppress them here.
        with contextlib.redirect_stderr(io.StringIO()):
            with contextlib.redirect_stdout(io.StringIO()):
                import cuequivariance_ops_torch  # noqa: F401
                import cuequivariance_torch as cuet  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _get_cueq_sh_module(
    l: int,
    normalize: bool,
    device_type: str,
    device_index: int,
) -> torch.nn.Module:
    """Build/cache a cuEquivariance spherical-harmonics module for one degree l."""
    import cuequivariance_torch as cuet

    device = torch.device(
        device_type
        if device_index < 0
        else f"{device_type}:{device_index}"
    )
    module = cuet.SphericalHarmonics([int(l)], normalize=normalize, use_fallback=False)
    return module.to(device=device)


def spherical_harmonics_real(
    l: int,
    vectors: torch.Tensor,
    *,
    normalize: bool,
    normalization: str,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute real spherical harmonics with optional cuEquivariance backend.

    Backends:
      - ``"e3nn"``: always uses ``e3nn.o3.spherical_harmonics``.
      - ``"cueq"``: requires ``cuequivariance_torch`` + ``cuequivariance_ops_torch``.
      - ``"auto"``: uses cuEquivariance when available, else falls back to e3nn.

    Note:
      cuEquivariance returns values scaled by ``sqrt(4π)`` relative to e3nn's
      ``normalization='integral'`` convention, so we rescale by ``1/sqrt(4π)``
      to preserve existing model semantics.
    """
    if backend not in {"auto", "e3nn", "cueq"}:
        raise ValueError(f"backend must be one of {{'auto','e3nn','cueq'}}, got {backend!r}")
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors must end with dim=3, got {tuple(vectors.shape)}")

    cueq_ready = _cueq_ops_available() and vectors.device.type == "cuda"
    use_cueq = backend == "cueq" or (backend == "auto" and cueq_ready)
    if backend == "cueq" and not cueq_ready:
        raise RuntimeError(
            "backend='cueq' requested, but cuequivariance kernels are unavailable "
            "for this device/runtime. Ensure CUDA device and cuequivariance-ops-torch are installed."
        )
    if not use_cueq or normalization != "integral":
        from e3nn import o3

        return o3.spherical_harmonics(
            int(l),
            vectors,
            normalize=normalize,
            normalization=normalization,
        )

    module = _get_cueq_sh_module(
        int(l),
        bool(normalize),
        vectors.device.type,
        -1 if vectors.device.index is None else int(vectors.device.index),
    )
    flat = vectors.reshape(-1, 3)
    y = module(flat).reshape(vectors.shape[:-1] + (2 * int(l) + 1,))
    y = y / math.sqrt(4.0 * math.pi)
    return y.to(dtype=vectors.dtype)
