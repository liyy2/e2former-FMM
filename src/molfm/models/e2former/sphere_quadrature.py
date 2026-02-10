"""Sphere quadrature schemes for the FMM plane-wave expansion.

The FMM factorization of the alphaf(r)Y operator approximates the kernel
j_l(r)Y_l(r) via a discrete sum over S^2:

    j_l(|r|) Y_l(r)  i^{-l} _s  w_s  Y_l(u_s)  exp(i u_s  r)

The quality of this approximation depends on the quadrature rule {u_s, w_s}
used to discretize the sphere integral.  This module provides four options
with different accuracy/complexity tradeoffs:

    fibonacci_sphere:     Fibonacci lattice  equal-weight quasi-uniform points.
                          Simple, no dependencies, good for quick prototyping.

    gauss_legendre_sphere: Tensor-product Gauss-Legendre  uniform- quadrature.
                          Exact for tensor-product integrands of moderate degree.

    e3nn_s2grid_sphere:   Band-limited Kostelec-Rockmore grid from e3nn.
                          Spectral accuracy for SH transforms up to degree lmax.

    lebedev_sphere:       Lebedev-Laikov quadrature (via SciPy).
                          Highest accuracy per point  exactly integrates
                          spherical polynomials of degree  order.

All functions return:
    dirs:    (S, 3)  unit direction vectors on the sphere
    weights: (S,)   quadrature weights normalized so  w_s = 1
                     (i.e., for the *average* over the sphere, not the integral)
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def fibonacci_sphere(num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate quasi-uniform points on S^2 using the Fibonacci lattice.

    The Fibonacci lattice distributes points on the sphere by:
      - Stratifying z (the polar coordinate) uniformly in [-1, 1]
      - Incrementing the azimuthal angle by the golden angle  = (3-5)
        to minimize alignment artifacts between rows

    This produces a deterministic, low-discrepancy point set with equal weights
    (1/S each).  The quality is comparable to Monte-Carlo integration but with
    better uniformity.  No external dependencies are needed.

    Args:
        num:    Number of points S to generate.
        device: Torch device for the output tensor.
        dtype:  Torch dtype for the output tensor.

    Returns:
        Tensor of shape (S, 3)  unit vectors on S^2.
    """
    if num <= 0:
        raise ValueError(f"num must be > 0, got {num}")

    i = torch.arange(num, device=device, dtype=dtype)

    # Stratify z uniformly in [-1, 1]. The offset +0.5 centers points within
    # their strata rather than placing them at boundaries.
    z = 1.0 - 2.0 * (i + 0.5) / float(num)

    # Convert z to the cylindrical radius r = sin(arccos(z)) = sqrt(1 - z^2).
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))

    # Golden angle in radians. Each successive point rotates by this amount
    # in azimuth, producing an irrational spacing that avoids grid artifacts.
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * i

    x = torch.cos(theta) * r
    y = torch.sin(theta) * r

    # Output layout (S, 3) matches the `s` direction index used in FMM einsums
    # (e.g., "nhd,ns,nhc->shdc" where s indexes quadrature directions).
    return torch.stack([x, y, z], dim=-1)


def _factorize_num_points(num: int) -> tuple[int, int]:
    """Find near-square factors (num_theta, num_phi) with num_theta * num_phi = num.

    For a tensor-product grid on S^2, we need to split the total point count
    into (num_theta, num_phi). This function finds the factorization closest to
    a square, preferring more azimuth points (phi) than polar points (theta)
    because:
      1. Uniform phi sampling is cheaper and better-conditioned.
      2. The plane-wave phase exp(i u_sr) oscillates more in azimuth than polar.

    Args:
        num: Total number of points (must be > 0).

    Returns:
        (num_theta, num_phi) with num_theta <= num_phi and product = num.
    """
    if num <= 0:
        raise ValueError(f"num must be > 0, got {num}")
    best_a, best_b = 1, num
    best_diff = num - 1
    limit = int(math.isqrt(num))
    for a in range(1, limit + 1):
        if num % a != 0:
            continue
        b = num // a
        diff = abs(b - a)
        if diff < best_diff:
            best_a, best_b = a, b
            best_diff = diff
    # Convention: num_theta <= num_phi (more azimuth samples).
    num_theta, num_phi = (best_a, best_b) if best_a <= best_b else (best_b, best_a)
    return int(num_theta), int(num_phi)


def gauss_legendre_sphere(
    num_points: int,
    *,
    num_theta: Optional[int] = None,
    num_phi: Optional[int] = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Tensor-product Gauss-Legendre  uniform- quadrature on S^2.

    This constructs a product quadrature rule:
      - Polar ():    Gauss-Legendre nodes in  = cos() on [-1, 1].
                       Exactly integrates polynomials of degree  2num_theta - 1 in .
      - Azimuthal (): Uniform trapezoid rule on [0, 2).
                       Exactly integrates trigonometric polynomials of degree < num_phi.

    The product rule is exact for functions that are separable into
    f()g() where f and g are within the polynomial degree limits.

    Args:
        num_points:  Total number of quadrature points S = num_theta  num_phi.
        num_theta:   Optional override for number of polar nodes.
        num_phi:     Optional override for number of azimuthal nodes.
        device:      Torch device for output tensors.
        dtype:       Torch dtype for output tensors.

    Returns:
        Tuple of (dirs, weights, num_theta, num_phi):
          - dirs:    (S, 3) unit direction vectors
          - weights: (S,) quadrature weights with sum = 1 (average convention)
          - num_theta, num_phi: the factorization used
    """
    import numpy as np

    # Auto-factorize if the user didn't specify both dimensions.
    if num_theta is None or num_phi is None:
        num_theta, num_phi = _factorize_num_points(num_points)
    if num_theta <= 0 or num_phi <= 0:
        raise ValueError(f"num_theta and num_phi must be > 0, got {num_theta}, {num_phi}")
    if num_theta * num_phi != num_points:
        raise ValueError(
            f"num_points must equal num_theta*num_phi, got {num_points} vs {num_theta}*{num_phi}"
        )

    # Gauss-Legendre nodes (_i) and weights (w_i) for _{-1}^{1} f() d.
    # These are computed via NumPy's eigenvalue-based algorithm.
    # sum(w_theta) = 2  (the measure of [-1,1]).
    mu, w_theta = np.polynomial.legendre.leggauss(int(num_theta))  #  in [-1,1]

    # Uniform azimuthal grid: _j = 2j/num_phi for j = 0, ..., num_phi-1.
    phi = (2.0 * np.pi) * (np.arange(int(num_phi), dtype=np.float64) / float(num_phi))

    # Convert from (, ) to Cartesian (x, y, z):
    #   x = sin()cos(),  y = sin()sin(),  z = cos() = 
    sin_theta = np.sqrt(np.clip(1.0 - mu * mu, 0.0, None))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x = sin_theta[:, None] * cos_phi[None, :]     # (num_theta, num_phi)
    y = sin_theta[:, None] * sin_phi[None, :]     # (num_theta, num_phi)
    z = mu[:, None] * np.ones((1, int(num_phi)), dtype=np.float64)  # (num_theta, num_phi)

    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (S, 3)

    # Combined weight for the *average* over S^2:
    #   _{S^2} f d / (4)  _s w_s f(u_s)
    # The product measure is:
    #   d = d d  =>  w_{ij} = w_theta_i  (2/num_phi) / (4)
    #                           = w_theta_i / (2num_phi)
    w = (w_theta[:, None] / (2.0 * float(num_phi))) * np.ones((1, int(num_phi)), dtype=np.float64)
    w = w.reshape(-1)  # (S,)

    # Convert to torch tensors on the requested device/dtype.
    dirs_t = torch.tensor(dirs, device=device, dtype=dtype)
    w_t = torch.tensor(w, device=device, dtype=dtype)
    return dirs_t, w_t, int(num_theta), int(num_phi)


def _choose_res_beta_alpha(
    num_points: int,
    *,
    min_res_beta: int,
    require_even_beta: bool = True,
) -> tuple[int, int]:
    """Choose (res_beta, res_alpha) factorization for the e3nn S2 grid.

    The e3nn Kostelec-Rockmore grid uses a (beta  alpha) layout where:
      - beta (): polar coordinate, requires even resolution for e3nn's internal FFT.
      - alpha (): azimuthal coordinate, higher is better for plane-wave accuracy.

    This function searches for the factorization of num_points that:
      1. Has even res_beta (required by e3nn's SH transform internals).
      2. Has res_beta >= min_res_beta (to support the required angular degree).
      3. Maximizes res_alpha (minimizes res_beta) to reduce azimuthal aliasing.

    Args:
        num_points:        Total grid points S = res_beta  res_alpha.
        min_res_beta:      Minimum polar resolution (typically 2*(l+1) for degree l).
        require_even_beta: If True (default), res_beta must be even.

    Returns:
        (res_beta, res_alpha) satisfying all constraints.
    """
    best: Optional[tuple[int, int]] = None
    for res_beta in range(1, num_points + 1):
        if num_points % res_beta != 0:
            continue
        if require_even_beta and (res_beta % 2 != 0):
            continue
        if res_beta < min_res_beta:
            continue
        res_alpha = num_points // res_beta
        if best is None:
            best = (res_beta, res_alpha)
            continue
        # Prefer the factorization with the largest azimuthal resolution,
        # since plane-wave phases exp(i ur) oscillate more in azimuth.
        if res_alpha > best[1]:
            best = (res_beta, res_alpha)
    if best is None:
        # Fallback: use _factorize_num_points and swap if needed for even beta.
        res_beta, res_alpha = _factorize_num_points(num_points)
        if require_even_beta and res_beta % 2 != 0:
            if res_alpha % 2 == 0:
                res_beta, res_alpha = res_alpha, res_beta
            else:
                raise ValueError(
                    f"Cannot find an even res_beta factorization for num_points={num_points}. "
                    "Try a different num_directions."
                )
        if res_beta < min_res_beta:
            raise ValueError(
                f"Cannot satisfy min_res_beta={min_res_beta} with num_points={num_points}. "
                "Try a larger num_directions."
            )
        best = (res_beta, res_alpha)
    return int(best[0]), int(best[1])


def e3nn_s2grid_sphere(
    num_points: int,
    *,
    l_required: int,
    res_beta: Optional[int] = None,
    res_alpha: Optional[int] = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Kostelec-Rockmore S2 grid from e3nn with spectral quadrature weights.

    This grid is used internally by e3nn for spherical harmonic transforms.
    It provides spectral accuracy: a function band-limited to degree lmax is
    reconstructed exactly if res_beta >= 2*(lmax + 1).

    The grid layout is (res_beta  res_alpha) where:
      - beta ():  polar angle, sampled at Driscoll-Healystyle nodes
      - alpha (): azimuthal angle, uniform on [0, 2)

    Quadrature weights are derived from e3nn's _quadrature_weights and rescaled
    to our "average over sphere" convention ( w_s = 1).

    Args:
        num_points:  Total grid points S = res_beta  res_alpha.
        l_required:  The SH degree l that the grid must support.
                     Requires res_beta >= 2*(l_required + 1).
        res_beta:    Optional override for polar resolution (must be even).
        res_alpha:   Optional override for azimuthal resolution.
        device:      Torch device.
        dtype:       Torch dtype.

    Returns:
        Tuple of (dirs, weights, res_beta, res_alpha).
    """
    from e3nn.o3 import _s2grid  # local import to avoid loading e3nn at module import time

    if res_beta is None or res_alpha is None:
        # e3nn constraint: lmax + 1 <= res_beta // 2  =>  res_beta >= 2*(l + 1)
        min_res_beta = 2 * (int(l_required) + 1)
        res_beta, res_alpha = _choose_res_beta_alpha(
            num_points, min_res_beta=min_res_beta, require_even_beta=True
        )
    if res_beta * res_alpha != num_points:
        raise ValueError(
            f"num_points must equal res_beta*res_alpha, got {num_points} vs {res_beta}*{res_alpha}"
        )
    if res_beta % 2 != 0:
        raise ValueError(f"res_beta must be even, got {res_beta}")
    if (int(l_required) + 1) > (res_beta // 2):
        raise ValueError(
            f"Grid too small for l_required={l_required}: need res_beta >= {2*(l_required+1)}, got {res_beta}"
        )

    # e3nn's s2_grid returns separable coordinate vectors for the grid:
    #   betas:  (res_beta,)  polar angle values in [0, ]
    #   alphas: (res_alpha,)  azimuthal angle values in [0, 2)
    betas, alphas = _s2grid.s2_grid(res_beta, res_alpha, dtype=torch.float64, device=device)
    beta = betas[:, None]  # (res_beta, 1)  broadcast for outer product
    alpha = alphas[None, :]  # (1, res_alpha)

    # Convert from (, ) to Cartesian (x, y, z):
    #   x = sin()cos(),  y = sin()sin(),  z = cos()
    sinb = torch.sin(beta)
    cosb = torch.cos(beta)
    z = cosb.expand(-1, int(res_alpha))
    dirs = torch.stack(
        [sinb * torch.cos(alpha), sinb * torch.sin(alpha), z],
        dim=-1,
    ).reshape(-1, 3)  # (S, 3)

    # Compute quadrature weights.
    # e3nn's _quadrature_weights returns the weight for each beta ring,
    # normalized for e3nn's internal SH transform convention. We rescale
    # to our "average" convention where  w_s = 1.
    qw = _s2grid._quadrature_weights(res_beta // 2, dtype=torch.float64, device=device)
    qw = qw * (res_beta**2) / float(res_alpha)  # (res_beta,)  rescaled
    # Expand each beta weight to all alpha values in that ring.
    w = qw[:, None].expand(res_beta, res_alpha).reshape(-1)  # (S,)

    # Convert to caller's dtype for efficient downstream computation.
    dirs = dirs.to(dtype=dtype)
    w = w.to(dtype=dtype)
    return dirs, w, int(res_beta), int(res_alpha)


# =====================================================================
# Lebedev-Laikov quadrature  highest accuracy per point
# =====================================================================
# Lebedev quadrature rules exactly integrate spherical polynomials of
# degree  order. They are rotationally invariant by construction (the
# point set has octahedral symmetry), making them ideal for integrating
# functions that contain low-degree spherical harmonics  exactly the
# case in the FMM plane-wave expansion.
#
# The tables below map between Lebedev orders (polynomial exactness degree)
# and the corresponding number of quadrature points. Only specific point
# counts are valid (they correspond to solved Lebedev designs).

_LEBEDEV_ORDERS = (
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
    35,
    41,
    47,
    53,
    59,
    65,
    71,
    77,
    83,
    89,
    95,
    101,
    107,
    113,
    119,
    125,
    131,
)
_LEBEDEV_POINTS = (
    6,      # order 3:   integrates up to degree 3 exactly
    14,     # order 5
    26,     # order 7
    38,     # order 9
    50,     # order 11
    74,     # order 13  <- a good default for low-l work
    86,     # order 15
    110,    # order 17
    146,    # order 19
    170,    # order 21
    194,    # order 23
    230,    # order 25
    266,    # order 27
    302,    # order 29  <- recommended for moderate accuracy
    350,    # order 31
    434,    # order 35
    590,    # order 41
    770,    # order 47
    974,    # order 53
    1202,   # order 59
    1454,   # order 65
    1730,   # order 71
    2030,   # order 77
    2354,   # order 83
    2702,   # order 89
    3074,   # order 95
    3470,   # order 101
    3890,   # order 107
    4334,   # order 113
    4802,   # order 119
    5294,   # order 125
    5810,   # order 131
)

# Bidirectional lookup: given a point count, find the Lebedev order.
_LEBEDEV_POINTS_TO_ORDER = dict(zip(_LEBEDEV_POINTS, _LEBEDEV_ORDERS))


def lebedev_sphere(
    num_points_or_order: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Lebedev-Laikov quadrature on S^2 using SciPy.

    Lebedev quadrature is the gold standard for numerical integration on the
    sphere. A rule of order p exactly integrates all spherical polynomials of
    degree  p, using the minimum possible number of points. The point sets
    have octahedral symmetry, which also helps with rotational invariance.

    This function accepts either:
      - A standard Lebedev **point count** (e.g., 74, 110, 302), or
      - A Lebedev **order** (e.g., 13, 17, 29) if it matches a supported order.

    SciPy's `lebedev_rule` returns weights that sum to 4 (the surface area of
    the unit sphere). We renormalize to sum = 1 to match the "average over sphere"
    convention used by all other quadrature functions in this module.

    Args:
        num_points_or_order: Either a supported point count or a supported order.
        device:              Torch device for output tensors.
        dtype:               Torch dtype for output tensors.

    Returns:
        Tuple of (dirs, weights, order, num_points):
          - dirs:    (S, 3) unit direction vectors
          - weights: (S,) quadrature weights with sum = 1
          - order:   the Lebedev polynomial exactness order
          - num_points: actual number of quadrature points S

    Raises:
        ValueError: If the requested size/order is not a supported Lebedev design.
    """
    from scipy.integrate import lebedev_rule  # type: ignore

    # Resolve the input to a Lebedev order.
    if num_points_or_order in _LEBEDEV_POINTS_TO_ORDER:
        # Input is a point count (e.g., 74 -> order 13).
        order = _LEBEDEV_POINTS_TO_ORDER[int(num_points_or_order)]
    elif int(num_points_or_order) in _LEBEDEV_ORDERS:
        # Input is directly a Lebedev order.
        order = int(num_points_or_order)
    else:
        supported = ", ".join(str(x) for x in _LEBEDEV_POINTS)
        raise ValueError(
            f"Unsupported Lebedev size/order: {num_points_or_order}. "
            f"Supported point counts: [{supported}] (or their corresponding orders)."
        )

    # SciPy returns:
    #   x: (3, m)  Cartesian coordinates of quadrature points
    #   w: (m,)  weights summing to 4 (surface area of unit sphere)
    x, w = lebedev_rule(order)

    # Normalize weights to sum = 1 (average convention, consistent with all
    # other quadrature routines in this module).
    w = w / w.sum()

    dirs = torch.tensor(x.T, device=device, dtype=dtype)     # (S, 3)  unit directions
    weights = torch.tensor(w, device=device, dtype=dtype)     # (S,)  average weights
    return dirs, weights, int(order), int(weights.numel())
