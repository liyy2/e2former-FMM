"""Exact (non-factorized) implementations of the alphaf(r)Y equivariant attention operator.

These classes compute the full O(N^2) pairwise geometry and attention matrices.
They serve as ground-truth references for validating the FMM approximation in
fmm_prototype.py.

Classes:
    AlphaFRYSphericalLinear:
        Exact operator with *linear* attention weights (positive feature map + normalization).
        Complexity: O(N^2).

    AlphaFRYSphericalVanilla:
        Backward-compatible alias for AlphaFRYSphericalLinear.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .fmm_utils import (
    infer_compute_dtype,
    positive_feature_map,
    prepare_radial_coeffs,
    prod,
    spherical_bessel_j,
    spherical_harmonics_real,
)


class AlphaFRYSphericalLinear(nn.Module):
    """Exact alphaf(r)Y operator with *linear* attention weights (O(N^2)).

    This is the "ground truth" reference used to validate the FMM approximation.
    It explicitly forms:
      - The NN linear-attention weight matrix [h,i,j]
      - The NN pairwise distance matrix and spherical harmonics
    and directly contracts over source nodes j.

    Mathematical definition:
        out[i, h, m, c] = _j  [i,h,j]  f_l(|r_ij|)  Y_{l,m}(r_ij)  v[j,h,c]

    where  uses linear (kernel) attention:
        [i,h,j] = <(q[i,h]), (k[j,h])> / _n <(q[i,h]), (k[n,h])>

    The FMM class (AlphaFRYSphericalFMM) approximates *this exact* computation
    by factorizing the kernel using the plane-wave expansion.

    Args:
        l:              Spherical harmonic degree (l=0: scalar, l=1: vector, ...).
        kappa:          Radial frequencies, shape (Q,).
        a:              Radial mixing coefficients. Shape (Q,), (H,Q), or (H,C,Q).
        feature_map:    Positive feature map for linear attention ("elu" or "softplus").
        normalization:  e3nn spherical harmonic normalization convention.
        sh_backend:     Spherical-harmonics backend ("auto" | "e3nn" | "cueq").
        eps:            Small constant for numerical stability in denominator.

    Shape symbols:
      - *shape: arbitrary leading dims, typically (B, num_nodes)
      - B: batch size, num_nodes: num nodes, N: prod(*shape)
      - H: num heads, d: per-head q/k dim, c: per-head value dim
      - l: SH degree, M = 2l+1: num m-components

    Complexity:
      - Time: O(H  N^2  d + H  N^2  M  c)
      - Memory: O(N^2) from explicit pairwise geometry and attention tensors
    """

    def __init__(
        self,
        l: int,
        kappa: torch.Tensor,
        a: torch.Tensor,
        *,
        feature_map: str = "elu",
        normalization: str = "integral",
        sh_backend: str = "auto",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        # --- Validate inputs ---
        if kappa.ndim != 1:
            raise ValueError(f"kappa must be 1D of shape (Q,), got {tuple(kappa.shape)}")
        if a.ndim not in (1, 2, 3):
            raise ValueError(
                f"a must have shape (Q,), (H,Q), or (H,C,Q), got {tuple(a.shape)}"
            )
        if a.shape[-1] != kappa.shape[0]:
            raise ValueError(
                f"a.shape[-1] must equal Q={kappa.shape[0]}, got a.shape={tuple(a.shape)}"
            )
        if l < 0:
            raise ValueError(f"l must be >= 0, got {l}")
        if sh_backend not in ("auto", "e3nn", "cueq"):
            raise ValueError(
                "sh_backend must be one of {'auto','e3nn','cueq'}, "
                f"got {sh_backend!r}"
            )

        self.l = int(l)
        self.feature_map = feature_map
        self.normalization = normalization
        self.sh_backend = sh_backend
        self.eps = float(eps)

        # Keep radial basis tensors as non-trainable buffers so they track
        # device, dtype, and state_dict with the module.
        self.register_buffer("kappa", kappa.detach().clone())
        self.register_buffer("a", a.detach().clone())

    def forward(
        self,
        pos: torch.Tensor,  # (*shape, 3)
        q: torch.Tensor,  # (*shape, d) or (*shape, H, d)
        k: torch.Tensor,  # (*shape, d) or (*shape, H, d)
        v: torch.Tensor,  # (*shape, c) or (*shape, H, c)
        node_mask: Optional[torch.Tensor] = None,  # (*shape,)
    ) -> torch.Tensor:
        """Compute the exact linear-attention alphaf(r)Y operator.

        This explicitly builds:
          1. The NN linear-attention matrix [h,i,j] = <(q_i), (k_j)> / Z_i
          2. The NN pairwise geometry: distances, spherical harmonics, Bessel functions.
          3. Contracts over j to produce equivariant output.

        This serves as the "ground truth" reference when debugging/validating the
        FMM approximation (AlphaFRYSphericalFMM), which approximates this same
        computation without forming the NN matrices.

        Args:
            pos:        Node positions, shape (*shape, 3).
            q:          Query features, shape (*shape, d) or (*shape, H, d).
            k:          Key features, shape (*shape, d) or (*shape, H, d).
            v:          Value features, shape (*shape, c) or (*shape, H, c).
            node_mask:  Optional boolean mask, shape (*shape,). True = valid node.

        Returns:
            Equivariant output of shape (*shape, H, 2l+1, c), or (*shape, 2l+1, c)
            if inputs were unheaded.
        """
        # =====================================================================
        # Phase 1: Input validation and reshaping
        # =====================================================================
        if pos.ndim < 2 or pos.shape[-1] != 3:
            raise ValueError(f"pos must be (*shape, 3), got {tuple(pos.shape)}")
        if q.shape[: pos.ndim - 1] != pos.shape[:-1] or k.shape[: pos.ndim - 1] != pos.shape[:-1]:
            raise ValueError("q and k must share the same leading *shape as pos")
        if v.shape[: pos.ndim - 1] != pos.shape[:-1]:
            raise ValueError("v must share the same leading *shape as pos")

        lead_shape = pos.shape[:-1]
        # Preserve graph boundaries in batched mode:
        # - last leading axis is node axis num_nodes
        # - all preceding leading axes are flattened into graph axis B
        graph_shape = lead_shape[:-1]
        B = int(prod(graph_shape)) if len(graph_shape) > 0 else 1
        num_nodes = int(lead_shape[-1])

        compute_dtype = infer_compute_dtype(pos.dtype)
        # Geometry-heavy math is performed in fp32 under AMP for numerical stability.
        pos_f = pos.reshape(B, num_nodes, 3).to(dtype=compute_dtype)

        q_flat = q.reshape((B, num_nodes) + q.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        k_flat = k.reshape((B, num_nodes) + k.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        v_flat = v.reshape((B, num_nodes) + v.shape[pos.ndim - 1 :])

        # Track whether caller passed headed tensors for correct output shape.
        had_heads = q_flat.ndim == 4

        # Normalize to headed layout (B, num_nodes, H, d). Unheaded inputs get H=1.
        if q_flat.ndim == 3:
            q_flat = q_flat.unsqueeze(2)  # (B, num_nodes, 1, d)
        if k_flat.ndim == 3:
            k_flat = k_flat.unsqueeze(2)

        if q_flat.ndim != 4 or k_flat.ndim != 4:
            raise ValueError("q and k must be (*shape,d) or (*shape,H,d)")
        if q_flat.shape[2:] != k_flat.shape[2:]:
            raise ValueError(
                f"q and k must match (H,d), got {q_flat.shape[2:]} vs {k_flat.shape[2:]}"
            )

        if v_flat.ndim == 3:
            v_flat = v_flat.unsqueeze(2)  # (B, num_nodes, 1, c)
        if v_flat.ndim != 4:
            raise ValueError("v must be (*shape,c) or (*shape,H,c)")

        H = q_flat.shape[2]
        if v_flat.shape[2] == 1 and H > 1:
            # Broadcast shared values across all heads.
            v_flat = v_flat.expand(-1, -1, H, -1)
        elif v_flat.shape[2] != H:
            raise ValueError(f"v must have head dim 1 or H={H}, got {v_flat.shape[2]}")
        C = v_flat.shape[-1]

        # =====================================================================
        # Phase 2: Positive feature maps and masking
        # =====================================================================
        # Linear attention requires (x) >= 0 so the denominator is well-behaved:
        #   Z_i = _j <(q_i), (k_j)>  >=  0
        phi_q = positive_feature_map(q_flat, kind=self.feature_map)  # (B, num_nodes, H, d)
        phi_k = positive_feature_map(k_flat, kind=self.feature_map)  # (B, num_nodes, H, d)

        if node_mask is not None:
            if node_mask.shape != lead_shape:
                raise ValueError(f"node_mask must be {tuple(lead_shape)}, got {tuple(node_mask.shape)}")
            m = node_mask.reshape(B, num_nodes).to(device=pos.device, dtype=compute_dtype)
            # Zero out features at padding positions. This ensures padded nodes
            # contribute nothing to numerator or denominator.
            phi_q = phi_q * m[:, :, None, None]
            phi_k = phi_k * m[:, :, None, None]
            v_flat = v_flat * m[:, :, None, None].to(dtype=v_flat.dtype)

        v_compute = v_flat.to(dtype=compute_dtype)

        # =====================================================================
        # Phase 3: Linear attention weights [h,i,j]  (O(N^2) explicit)
        # =====================================================================
        # [b,h,i,j] = <(q_i)[b,h], (k_j)[b,h]> / _n <(q_i)[b,h], (k_n)[b,h]>
        #
        # Unlike the FMM path which avoids the NN matrix, here we form it
        # explicitly as the ground-truth reference.
        # Einsum: _q(b,i,h,d)  _k(b,j,h,d) -> logits(b,h,i,j)  [reduce d]
        logits = torch.einsum("bihd,bjhd->bhij", phi_q, phi_k)  # (B, H, num_nodes, num_nodes)
        # Per-(head, query) normalization. eps prevents div-by-zero at masked queries.
        denom = logits.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, H, num_nodes, 1)
        alpha = logits / denom  # (B, H, num_nodes, num_nodes) normalized attention weights

        # =====================================================================
        # Phase 4: Pairwise geometry  displacements, distances, SH, Bessel
        # =====================================================================
        # Build all pairwise displacement vectors per graph: r_ij = r_i - r_j.
        r_ij = pos_f[:, :, None, :] - pos_f[:, None, :, :]  # (B, num_nodes, num_nodes, 3)
        dist = torch.linalg.vector_norm(r_ij, dim=-1)  # (B, num_nodes, num_nodes)

        # Replace zero displacements on the diagonal with a safe direction
        # to avoid NaN gradients in e3nn's spherical_harmonics(normalize=True).
        r_ij_safe = r_ij.clone()
        diag = torch.arange(num_nodes, device=pos.device)
        r_ij_safe[:, diag, diag] = torch.tensor(
            [1.0, 0.0, 0.0],
            device=pos.device,
            dtype=compute_dtype,
        )

        # Angular factor: Y_{l,m}(r_ij)  real spherical harmonics of the displacement direction.
        Y = spherical_harmonics_real(
            self.l,
            r_ij_safe,
            normalize=True,
            normalization=self.normalization,
            backend=self.sh_backend,
        )  # (B, num_nodes, num_nodes, 2l+1)

        # Radial factor: f_l(r) = _q a_q  j_l(_q  r)
        # Spherical Bessel functions of the first kind form a natural basis for
        # radial functions arising from the Helmholtz equation in spherical coords.
        x = dist[..., None] * self.kappa.to(dist.dtype)  # (B, num_nodes, num_nodes, Q)
        jl = spherical_bessel_j(self.l, x, eps=self.eps)  # (B, num_nodes, num_nodes, Q)

        # Prepare radial mixing coefficients for the appropriate broadcast mode.
        a_coeff, a_mode = prepare_radial_coeffs(
            self.a,
            num_kappa=int(self.kappa.shape[0]),
            num_heads=H,
            value_dim=C,
            device=pos.device,
            dtype=jl.dtype,
        )

        # =====================================================================
        # Phase 5: Combine attention  geometry  values, contract over j
        # =====================================================================
        # out[b,i,h,m,c] = _j  [b,h,i,j]  f(r_ij)  Y_{l,m}(r_ij)  v[b,j,h,c]
        # The form of f depends on a_mode:
        if a_mode == "scalar":
            # f(r_ij) is the same scalar for all heads and channels.
            f = torch.einsum("bijq,q->bij", jl, a_coeff)  # (B, num_nodes, num_nodes)
            w = alpha * f[:, None, :, :]  # (B, H, num_nodes, num_nodes)
            # Contract: w(b,h,i,j)  Y(b,i,j,m)  v(b,j,h,c) -> out(b,i,h,m,c)
            out = torch.einsum("bhij,bijm,bjhc->bihmc", w, Y, v_compute)
        elif a_mode == "head":
            # f depends on head h: each head has its own radial profile.
            f = torch.einsum("bijq,hq->bhij", jl, a_coeff)  # (B, H, num_nodes, num_nodes)
            w = alpha * f
            out = torch.einsum("bhij,bijm,bjhc->bihmc", w, Y, v_compute)
        else:
            # f depends on both head h and value channel c (most expressive).
            f = torch.einsum("bijq,hcq->bhijc", jl, a_coeff)  # (B, H, num_nodes, num_nodes, C')
            if f.shape[-1] == 1 and C > 1:
                f = f.expand(-1, -1, -1, -1, C)
            w = alpha.unsqueeze(-1) * f  # (B, H, num_nodes, num_nodes, C)
            out = torch.einsum("bhijc,bijm,bjhc->bihmc", w, Y, v_compute)

        # =====================================================================
        # Phase 6: Restore original shape and return
        # =====================================================================
        out = out.to(dtype=v.dtype)
        out = out.reshape(lead_shape + (H, 2 * self.l + 1, v_flat.shape[-1]))
        if not had_heads and q.ndim == pos.ndim and v.ndim == pos.ndim:
            out = out.squeeze(-3)  # remove singleton head dim for unheaded inputs
        return out


class AlphaFRYSphericalVanilla(AlphaFRYSphericalLinear):
    """Backward-compatible alias for AlphaFRYSphericalLinear.

    The name "vanilla" was used in earlier versions of this code to mean
    "linear attention baseline" (as opposed to softmax or FMM). New code
    should use AlphaFRYSphericalLinear directly.
    """

    def __init__(
        self,
        l: int,
        kappa: torch.Tensor,
        a: torch.Tensor,
        *,
        feature_map: str = "elu",
        normalization: str = "integral",
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            l=l,
            kappa=kappa,
            a=a,
            feature_map=feature_map,
            normalization=normalization,
            eps=eps,
        )
