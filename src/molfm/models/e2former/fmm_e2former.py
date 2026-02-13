# -*- coding: utf-8 -*-
"""Node-only FMM attention for E2Former.

This module provides `E2AttentionNodeFMM`, an attention layer that replaces
E2Former's original edge-based neighbor attention with a **global, node-only**
mechanism powered by the Fast Multipole Method (FMM).

High-level design:
    Standard E2Former attention constructs attention weights from *edges*
    (neighbor pairs within a cutoff radius) and uses Clebsch-Gordan (CG) tensor
    products (via Wigner-6j symbols) to couple spherical harmonics. This module
    takes a fundamentally different approach:

    1. **No edges**: Attention is global over all nodes within each graph.
       The FMM plane-wave expansion approximates the pairwise radial/angular
       kernel, avoiding explicit construction of NN edge tensors.

    2. **Value irreps are explicit**: Instead of operating on pre-contracted
       edge messages, the module keeps the per-node value irreps v_j^(lambda)
       as separate (lmax+1)^2 components and couples them *after* FMM
       aggregation using e3nn's FullTensorProduct.

    3. **CG coupling via tensor product**: The angular coupling between the FMM
       output (indexed by ell from the spherical Bessel/Y_l expansion) and the
       value irreps (indexed by lambda from the node features) is performed by
       a precomputed e3nn FullTensorProduct:
           output_L = sum_{ell,lambda} CG(ell, lambda -> L) * FMM_ell * v_lambda
       This replaces the Wigner-6j pathway used in standard E2Former.

Data flow summary:
    Input:  node_pos (N, 3), node_irreps (N, (lmax+1)^2, C), batch (N,)
     -> Pack nodes into padded graphs: (B, max_nodes, ...)
     -> Q, K projections via SO3_Linear2Scalar (irreps -> scalar per head)
     -> V = node_irreps reshaped into (B, max_nodes, H, (lmax+1)^2, Cv)
     -> FMM multi-L core: for each degree ell, produces
            out_ell[i,h,m_ell,lambda,Cv] via linear-attention + plane-wave
     -> CG coupling: contract (m_ell, lambda) with precomputed TP matrix
            to produce out_L[i,h,m_L,Cv] for each output degree L
     -> Average over coupling paths, unpack back to flat node indexing
    Output: (N, (lmax+1)^2, C)

Notation / index conventions used in this file:
    N:     total number of nodes (flat, across all graphs)
    B:     number of graphs in the batch
    max_nodes: maximum number of nodes in any graph (padding dimension)
    H:     number of attention heads (num_attn_heads)
    d:     per-head query/key dimension (attn_scalar_head)
    Cv:    per-head value dimension used inside the FMM core. By default
           Cv = scalar_dim / H, but this module can optionally bottleneck
           values via `fmm_value_head_dim` for speed.
    C:     total feature channels (scalar_dim = H * Cv)
    ell:   degree index from the FMM spherical Bessel expansion (0..lmax)
    lambda: degree index from the input value irreps (0..lmax)
    L:     output degree after CG coupling (0..lmax, selected from TP output)
    m_ell: order components for degree ell, dim = 2*ell+1
    Q:     number of irrep components = (lmax+1)^2
    R_total: total output dimension of the TP = dim(irreps_ell x irreps_lambda)
"""
import warnings

import e3nn
import torch
from e3nn import o3
from torch import nn

from .fmm_prototype import AlphaFRYSphericalFMMMultiL
from .fmm_utils import _cueq_ops_available
from .module_utils import SO3_Linear2Scalar_e2former


class E2AttentionNodeFMM(nn.Module):
    """Node-only FMM attention that replaces E2Former's edge-based attention.

    This module intentionally avoids edge-based neighbor attention construction and
    runs global attention per graph using node positions and node irreps only.
    It follows the node-wise moment formulation in `FMM/fmm.tex`, keeps the
    value irreps `v_j^(lambda)` explicit, and performs CG coupling through e3nn
    full tensor products (not Wigner-6j tensor-product modules).

    Architecture overview:
        1. Q/K projection: SO3_Linear2Scalar converts full irreps to scalar queries/keys.
        2. FMM core (AlphaFRYSphericalFMMMultiL): Computes linear-attention-weighted
           spherical Bessel/harmonic expansion for ALL degrees ell simultaneously,
           sharing geometry (phase tensors, moments) across degrees.
        3. CG coupling: A precomputed e3nn FullTensorProduct matrix couples the FMM
           output degree (ell) with the value irreps degree (lambda) to produce
           the final equivariant output at each degree L.

    The constructor signature deliberately mirrors the original E2Former attention
    module's interface (accepting and discarding many kwargs) so this module can
    be used as a drop-in replacement via config changes.

    Args:
        irreps_node_input: e3nn Irreps string or object specifying the input node
            feature representation (e.g., "256x0e+64x1e+32x2e"). The multiplicity
            of l=0 gives scalar_dim (total channels C), and the highest l gives lmax.
        num_attn_heads: Number of attention heads H.
        attn_scalar_head: Per-head query/key dimension d.
        tp_type: String controlling the CG tensor-product backend. Recognized tags
            (case-insensitive, '+'-separated):
            - "tp_e3nn" / "e3nn_tp": force the e3nn backend
            - "tp_cueq" / "cueq_tp": force the cuEquivariance backend
            - anything else (including "fmm-node"): auto-detect best available
        fmm_num_kappa: Number of radial frequencies used by the FMM core.
        fmm_kappa_min: Minimum radial frequency.
        fmm_kappa_max: Maximum radial frequency. A tighter band improves
            finite-direction equivariance at unchanged runtime.
        fmm_value_head_dim: Optional value bottleneck per head (Cv). If > 0 and
            different from (scalar_dim / H), values are projected down before the
            FMM core and projected back afterwards (bias-free, equivariant).
        fmm_learnable_radial_coeffs: If True, optimize spectral radial-mixture
            coefficients during training.
        fmm_radial_coeffs_mode: Layout of radial coefficients:
            "per_l_head" | "per_l_shared" | "head" | "shared".
        fmm_radial_init_scale: Initial coefficient scale (small values are safer).
        fmm_radial_low_kappa_bias: Exponential low-frequency preference at init.
        (all other kwargs are accepted for interface compatibility but discarded)
    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="fmm-node",
        attn_type="fmm-node",
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        norm_layer="identity",
        fmm_num_kappa: int = 8,
        fmm_kappa_min: float = 1.0,
        fmm_kappa_max: float = 1.4,
        fmm_num_directions: int = 25,
        fmm_kappa_chunk_size: int = 0,
        fmm_compute_dtype: str | None = "auto",
        fmm_value_head_dim: int = 0,
        fmm_learnable_radial_coeffs: bool = True,
        fmm_radial_coeffs_mode: str = "per_l_head",
        fmm_radial_init_scale: float = 0.05,
        fmm_radial_low_kappa_bias: float = 2.0,
        **kwargs,
    ):
        super().__init__()

        # Parse which CG tensor-product backend to use (e3nn vs cuEquivariance)
        # before we discard the tp_type string.
        tp_backend_request = self._parse_tp_backend_request(tp_type)

        # Discard all parameters that exist only for interface compatibility with
        # the original E2Former attention module. This module does not use edges,
        # distance polynomials, RoPE, atom types, or dropout -- they are all
        # superseded by the FMM formulation.
        del (
            attn_weight_input_dim,
            irreps_head,
            alpha_drop,
            rescale_degree,
            nonlinear_message,
            proj_drop,
            tp_type,
            attn_type,
            add_rope,
            layer_id,
            irreps_origin,
            neighbor_weight,
            atom_type_cnt,
            norm_layer,
            kwargs,
        )

        # =====================================================================
        # 1. Parse input irreps to extract key dimensions
        # =====================================================================
        # irreps_node_input describes the SO(3) representation at each node.
        # Example: "256x0e+64x1e+32x2e" means
        #   - 256 channels at l=0 (scalars)
        #   - 64 channels at l=1 (vectors)
        #   - 32 channels at l=2 (rank-2 tensors)
        # The E2Former convention stores these as (N, (lmax+1)^2, C) where C is
        # the multiplicity of l=0 (scalar_dim), and each (l,m) component shares
        # the same channel dimension C.
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.lmax = self.irreps_node_input[-1][1][0]   # highest angular momentum degree
        self.scalar_dim = self.irreps_node_input[0][0]  # total feature channels C (= multiplicity at l=0)
        self.num_attn_heads = int(num_attn_heads)       # H
        self.attn_scalar_head = int(attn_scalar_head)   # d (per-head Q/K feature dim)

        if self.num_attn_heads <= 0:
            raise ValueError(f"num_attn_heads must be > 0, got {self.num_attn_heads}")
        if self.scalar_dim <= 0:
            raise ValueError(f"scalar_dim must be > 0, got {self.scalar_dim}")
        if self.scalar_dim % self.num_attn_heads != 0:
            raise ValueError(
                "For full lambda-coupled FMM attention, scalar_dim must be divisible "
                f"by num_attn_heads. Got scalar_dim={self.scalar_dim}, heads={self.num_attn_heads}."
            )

        # =====================================================================
        # 2. Build Q/K projections (irreps -> scalar per head)
        # =====================================================================
        # Each node's full SO(3) irreps are "scalarized" into per-head query/key
        # vectors. SO3_Linear2Scalar_e2former performs:
        #   (N, (lmax+1)^2, C) -> (N, H*d)
        # using a degree-wise linear transform followed by per-degree inner products
        # and a final MLP. This collapses angular information into scalars suitable
        # for the attention dot product.
        # Default V head dim follows standard attention: Cv = C / H.
        # Optionally shrink Cv via a bias-free linear bottleneck (equivariant).
        value_head_dim_in = self.scalar_dim // self.num_attn_heads
        value_head_dim_req = int(fmm_value_head_dim)
        self.value_head_dim = value_head_dim_in if value_head_dim_req <= 0 else value_head_dim_req
        self.v_proj: nn.Linear | None = None
        self.out_proj: nn.Linear | None = None
        if self.value_head_dim != value_head_dim_in:
            # Note: bias would break equivariance for l>0 components (would introduce
            # fixed tensors in coefficient space). Keep these strictly linear.
            self.v_proj = nn.Linear(
                self.scalar_dim,
                self.num_attn_heads * self.value_head_dim,
                bias=False,
            )
            self.out_proj = nn.Linear(
                self.num_attn_heads * self.value_head_dim,
                self.scalar_dim,
                bias=False,
            )
        self.q_proj = SO3_Linear2Scalar_e2former(
            self.scalar_dim,                              # C_in = scalar_dim
            self.num_attn_heads * self.attn_scalar_head,  # C_out = H * d
            lmax=self.lmax,
        )
        self.k_proj = SO3_Linear2Scalar_e2former(
            self.scalar_dim,
            self.num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
        )

        # =====================================================================
        # 3. Build the FMM core (multi-degree, shared geometry)
        # =====================================================================
        if int(fmm_num_kappa) <= 0:
            raise ValueError(f"fmm_num_kappa must be > 0, got {fmm_num_kappa}")
        if float(fmm_kappa_max) < float(fmm_kappa_min):
            raise ValueError(
                f"fmm_kappa_max must be >= fmm_kappa_min, got "
                f"{fmm_kappa_max} < {fmm_kappa_min}"
            )
        if float(fmm_radial_init_scale) < 0.0:
            raise ValueError(
                f"fmm_radial_init_scale must be >= 0, got {fmm_radial_init_scale}"
            )
        if float(fmm_radial_low_kappa_bias) < 0.0:
            raise ValueError(
                "fmm_radial_low_kappa_bias must be >= 0, got "
                f"{fmm_radial_low_kappa_bias}"
            )
        # Radial basis: f_l(r) = sum_q a_q * j_l(kappa_q * r)
        # The compact default band (1.0..1.4) significantly reduces directional
        # quadrature aliasing (better equivariance) while keeping compute cost unchanged.
        num_kappa = int(fmm_num_kappa)
        kappa = torch.linspace(
            float(fmm_kappa_min),
            float(fmm_kappa_max),
            num_kappa,
            dtype=torch.float32,
        )
        # Learnable spectral radial mixture. Initialize coefficients to be:
        # - small amplitude (stable optimization at startup),
        # - biased toward low-kappa components (better short/long separation).
        a, a_per_l = self._build_initial_radial_coeffs(
            lmax=self.lmax,
            num_heads=self.num_attn_heads,
            kappa=kappa,
            mode=fmm_radial_coeffs_mode,
            init_scale=float(fmm_radial_init_scale),
            low_kappa_bias=float(fmm_radial_low_kappa_bias),
        )

        self.fmm_compute_dtype = fmm_compute_dtype

        # AlphaFRYSphericalFMMMultiL computes the FMM expansion for ALL degrees
        # l=0..lmax simultaneously, sharing expensive work (phase computation,
        # moment formation, query projection) and only branching at the final
        # sphere projection step (where W_l = w * Y_l(u) differs per degree).
        #
        # Configuration choices:
        #   - num_directions=25: 5x5 Gauss-Legendre product grid on S^2
        #   - sphere="gauss_legendre": tensor-product quadrature (accurate, GPU-friendly)
        #   - phase_mode="trig": use real cos/sin arithmetic (avoids complex tensors)
        #   - kappa_chunk_size=0: auto-chunk with OOM backoff for CUDA
        #   - promote_half_precision=False: keep native precision for throughput
        #   - optimize_low_precision_sphere=True: auto-switch sphere rule for low precision
        self.fmm_multi_l = AlphaFRYSphericalFMMMultiL(
            l_values=list(range(self.lmax + 1)),
            kappa=kappa,
            a=a,
            num_directions=int(fmm_num_directions),
            sphere="gauss_legendre",
            phase_mode="trig",
            kappa_chunk_size=int(fmm_kappa_chunk_size),
            promote_half_precision=False,
            optimize_low_precision_sphere=True,
            a_per_l=a_per_l,
            learnable_a=bool(fmm_learnable_radial_coeffs),
        )

        # =====================================================================
        # 4. Build CG coupling tensor product (precomputed at init time)
        # =====================================================================
        # The FMM core returns, for each degree ell:
        #   out_ell[i, h, m_ell, lambda_component, Cv]
        # where m_ell indexes the (2*ell+1) spherical harmonic components from
        # the plane-wave expansion, and lambda_component indexes the (lmax+1)^2
        # components of the value irreps v_j^(lambda).
        #
        # To produce equivariant output at degree L, we need Clebsch-Gordan coupling:
        #   output_L[m_L] = sum_{ell, lambda} CG(ell,m_ell; lambda,m_lambda | L,m_L)
        #                                     * out_ell[m_ell, m_lambda]
        #
        # We precompute this coupling as a matrix via e3nn's FullTensorProduct.
        # The "right" projection matrix R encodes all CG coefficients:
        #   R[lambda_component, ell_component, output_component]
        # where output_component indexes into the TP output irreps.

        # Build irreps for lambda (value side) and ell (FMM/Bessel side).
        # Both span l=0..lmax with multiplicity 1 each:
        #   irreps_lambda_all = "1x0e + 1x1e + ... + 1xlmax_e"
        #   irreps_ell_all    = "1x0e + 1x1e + ... + 1xlmax_e"
        # Their direct sum has dimension (lmax+1)^2 = sum_{l=0}^{lmax} (2l+1).
        self.irreps_lambda_all = o3.Irreps(
            "+".join(f"1x{lam}e" for lam in range(self.lmax + 1))
        )
        self.num_irrep_components = self.irreps_lambda_all.dim  # = (lmax+1)^2
        expected_components = (self.lmax + 1) ** 2
        if self.num_irrep_components != expected_components:
            raise RuntimeError(
                "Unexpected lambda irreps dimension: "
                f"{self.num_irrep_components} vs expected {expected_components}."
            )

        self.irreps_ell_all = o3.Irreps(
            "+".join(f"1x{ell}e" for ell in range(self.lmax + 1))
        )
        if self.irreps_ell_all.dim != expected_components:
            raise RuntimeError(
                "Unexpected ell irreps dimension: "
                f"{self.irreps_ell_all.dim} vs expected {expected_components}."
            )

        # Compute the full tensor product:
        #   irreps_ell (P dim) x irreps_lambda (Q dim) -> irreps_out (R_total dim)
        #
        # For lmax=2:
        #   (1x0e + 1x1e + 1x2e) x (1x0e + 1x1e + 1x2e) produces:
        #   all irreps L satisfying |ell-lambda| <= L <= ell+lambda for each (ell,lambda) pair.
        #
        # The "right" matrix precomputes the contraction:
        #   right_all[Q, P, R_total] such that
        #   output[R_total] = sum_{p,q} left[p] * right_all[q, p, R_total]
        # This avoids runtime CG evaluation -- the coupling is a single GEMM.
        tp_all = o3.FullTensorProduct(
            self.irreps_ell_all,
            self.irreps_lambda_all,
            compile_right=True,
        )
        y_basis = torch.eye(self.num_irrep_components, dtype=torch.float32)

        # Try to build the right-projection via cuEquivariance (GPU-accelerated CG),
        # falling back to e3nn's CPU implementation if unavailable.
        right_all: torch.Tensor | None = None
        resolved_backend = "e3nn"
        if tp_backend_request in ("auto", "cueq"):
            right_all = self._build_tp_right_from_cueq(
                irreps_ell=self.irreps_ell_all,
                irreps_lambda=self.irreps_lambda_all,
            )
            if right_all is not None:
                resolved_backend = "cueq"
            elif tp_backend_request == "cueq":
                raise RuntimeError(
                    "tp_backend='cueq' requested but cuequivariance TP is unavailable."
                )
        if right_all is None:
            # e3nn fallback: evaluate the TP on the identity basis to get the
            # right-projection matrix. Shape: (Q, P, R_total)
            right_all = tp_all.right(y_basis).to(dtype=torch.float32)
            resolved_backend = "e3nn"
        self.tp_backend = resolved_backend

        # =====================================================================
        # 5. Post-process the TP right-projection for efficient forward pass
        # =====================================================================
        # Normalize by path weights so that all CG paths contribute equally
        # regardless of e3nn's internal normalization convention.
        path_weight = torch.ones(len(tp_all.irreps_out), dtype=torch.float32)
        for ins in tp_all.instructions:
            path_weight[ins.i_out] = float(ins.path_weight)

        # Build an index of which output irreps blocks correspond to which
        # output degree L. Each block is (out_l, start_offset, end_offset, multiplicity).
        # Multiple (ell, lambda) pairs can couple to the same L, contributing
        # separate "paths" that will be summed in forward().
        self.tp_out_blocks: list[tuple[int, int, int, int]] = []
        coupling_count = torch.zeros(self.lmax + 1, dtype=torch.float32)
        offset = 0
        for out_idx, (mul, ir) in enumerate(tp_all.irreps_out):
            dim = int(mul * ir.dim)
            # Remove the e3nn path weight from the precomputed matrix so we
            # control normalization ourselves (divide by coupling_count later).
            right_all[:, :, offset : offset + dim].div_(path_weight[out_idx])
            out_l = int(ir.l)
            if out_l <= self.lmax:
                self.tp_out_blocks.append((out_l, offset, offset + dim, int(mul)))
                coupling_count[out_l] += float(mul)
            offset += dim

        self.tp_out_dim = int(tp_all.irreps_out.dim)  # R_total

        # Register the full right-projection as a buffer (not a parameter).
        self.register_buffer("tp_right_all", right_all.contiguous())

        # Flatten the full coupling tensor once:
        #   right_all[Q, P, R_total] -> tp_right_flat_all[P*Q, R_total]
        # where:
        #   P = sum_ell (2*ell+1) = (lmax+1)^2  (FMM spherical components)
        #   Q = (lmax+1)^2                      (value irrep components)
        #
        # This supports a single fused GEMM in forward() across all ell blocks.
        self.num_ell_components = int(self.irreps_ell_all.dim)
        tp_right_flat_all = (
            right_all.permute(1, 0, 2)  # (P, Q, R_total)
            .contiguous()
            .view(self.num_ell_components * self.num_irrep_components, self.tp_out_dim)
            # (P*Q, R_total)
        )
        self.register_buffer("tp_right_flat_all", tp_right_flat_all.contiguous())

        # coupling_count[L] = number of (ell, lambda) paths that produce degree L.
        # Used to average (not just sum) contributions in forward, preventing
        # higher-connectivity degrees from having disproportionately large outputs.
        self.register_buffer("coupling_count", coupling_count.clamp_min(1.0))

    # =================================================================
    # Helper methods
    # =================================================================

    @staticmethod
    def _pack_by_graph(
        node_pos: torch.Tensor,
        node_irreps: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Convert flat node tensors into padded per-graph batches.

        E2Former (and PyG in general) stores all nodes from all graphs in a single
        flat tensor with a `batch` vector indicating which graph each node belongs to.
        The FMM core, however, needs a regular (B, max_nodes, ...) layout so that
        the plane-wave expansion and moment aggregation operate within each graph
        independently (no cross-graph information leakage).

        This method pads each graph to `max_nodes` (the size of the largest graph)
        and returns a boolean mask distinguishing real nodes from padding.

        Args:
            node_pos:    (N, 3) flat positions for all nodes across all graphs.
            node_irreps: (N, (lmax+1)^2, C) flat irreps features for all nodes.
            batch:       (N,) integer graph-membership vector (0-indexed).

        Returns:
            packed_pos:    (B, max_nodes, 3) padded positions (zeros for padding).
            packed_irreps: (B, max_nodes, (lmax+1)^2, C) padded irreps.
            packed_mask:   (B, max_nodes) bool mask, True = real node.
            pack_info:     Opaque packing metadata consumed by _unpack_by_graph.
                           If the input `batch` is nondecreasing (common in this repo),
                           we use a fast scatter-based packing path and store
                           (mode, batch, idx_in_graph). Otherwise we use an
                           argsort+scatter packing path (no Python per-graph loops)
                           and store (mode, perm, batch_sorted, idx_in_graph_sorted).
        """
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        counts = torch.bincount(batch, minlength=num_graphs)
        max_nodes = int(counts.max().item()) if counts.numel() > 0 else 0

        packed_pos = node_pos.new_zeros((num_graphs, max_nodes, 3))
        packed_irreps = node_irreps.new_zeros(
            (num_graphs, max_nodes, node_irreps.shape[1], node_irreps.shape[2])
        )
        packed_mask = torch.zeros(
            (num_graphs, max_nodes), dtype=torch.bool, device=node_pos.device
        )

        # Fast path: in this repo, `batch` is typically nondecreasing because we
        # flatten padded (B,L,...) tensors with a boolean mask. Avoiding per-graph
        # `nonzero` calls removes a noticeable CPU/GPU synchronization bottleneck.
        batch_sorted = True
        if batch.numel() > 1:
            batch_sorted = bool((batch[1:] >= batch[:-1]).all().item())
        if batch_sorted:
            # start[g] = starting flat index of graph g in the nondecreasing layout
            start = torch.zeros((num_graphs,), device=batch.device, dtype=torch.long)
            if num_graphs > 1:
                start[1:] = counts.cumsum(dim=0)[:-1]
            idx_in_graph = torch.arange(batch.numel(), device=batch.device, dtype=torch.long)
            idx_in_graph = idx_in_graph - start[batch]
            packed_pos[batch, idx_in_graph] = node_pos
            packed_irreps[batch, idx_in_graph] = node_irreps
            packed_mask[batch, idx_in_graph] = True
            pack_info = ("sorted_batch", batch, idx_in_graph)
            return packed_pos, packed_irreps, packed_mask, pack_info

        # Fallback: general (unsorted) batch layouts.
        #
        # Avoid a Python `for graph_id in range(num_graphs)` loop (which becomes
        # expensive when there are many small graphs). We sort nodes by graph id,
        # then scatter into the dense (B, max_nodes, ...) tensors.
        perm = batch.argsort()
        batch_s = batch[perm]
        pos_s = node_pos[perm]
        irreps_s = node_irreps[perm]

        start = torch.zeros((num_graphs,), device=batch.device, dtype=torch.long)
        if num_graphs > 1:
            start[1:] = counts.cumsum(dim=0)[:-1]
        idx_in_graph_s = torch.arange(batch_s.numel(), device=batch.device, dtype=torch.long)
        idx_in_graph_s = idx_in_graph_s - start[batch_s]

        packed_pos[batch_s, idx_in_graph_s] = pos_s
        packed_irreps[batch_s, idx_in_graph_s] = irreps_s
        packed_mask[batch_s, idx_in_graph_s] = True
        pack_info = ("argsort_batch", perm, batch_s, idx_in_graph_s)
        return packed_pos, packed_irreps, packed_mask, pack_info

    @staticmethod
    def _unpack_by_graph(
        packed_output: torch.Tensor,
        pack_info: tuple,
        n_nodes: int,
    ) -> torch.Tensor:
        """Scatter padded per-graph output back to flat node indexing.

        Inverse of _pack_by_graph: takes (B, max_nodes, ...) output and writes
        each graph's valid nodes back into a flat (N, ...) tensor using the
        index mapping saved during packing.

        Args:
            packed_output: (B, max_nodes, (lmax+1)^2, C) padded output.
            pack_info:     packing metadata from _pack_by_graph.
            n_nodes:       total number of nodes N (to allocate output).

        Returns:
            (N, (lmax+1)^2, C) flat output tensor.
        """
        mode = str(pack_info[0])
        if mode == "sorted_batch":
            _, batch, idx_in_graph = pack_info
            if int(batch.numel()) != int(n_nodes):
                raise ValueError(
                    f"pack_info batch length {int(batch.numel())} must match n_nodes={int(n_nodes)}"
                )
            return packed_output[batch, idx_in_graph]

        if mode == "argsort_batch":
            _, perm, batch_s, idx_in_graph_s = pack_info
            if int(perm.numel()) != int(n_nodes):
                raise ValueError(
                    f"pack_info perm length {int(perm.numel())} must match n_nodes={int(n_nodes)}"
                )
            out_sorted = packed_output[batch_s, idx_in_graph_s]
            out = out_sorted.new_empty((n_nodes,) + out_sorted.shape[1:])
            out[perm] = out_sorted
            return out

        raise ValueError(f"Unknown pack_info mode {mode!r}")

    @staticmethod
    def _l_slice(order_l: int) -> tuple[int, int]:
        """Return the (start, end) index range for degree l in a packed (lmax+1)^2 vector.

        In the packed irreps layout, degree l occupies indices [l^2, (l+1)^2).
        For example:
            l=0: [0, 1)    (1 component: m=0)
            l=1: [1, 4)    (3 components: m=-1,0,+1)
            l=2: [4, 9)    (5 components: m=-2,-1,0,+1,+2)
        """
        start = order_l * order_l
        end = (order_l + 1) * (order_l + 1)
        return start, end

    @staticmethod
    def _parse_tp_backend_request(tp_type: str) -> str:
        """Extract the desired CG tensor-product backend from the tp_type config string.

        The tp_type string can contain '+'-separated tags. Recognized tags:
            "tp_e3nn" or "e3nn_tp" -> force e3nn
            "tp_cueq" or "cueq_tp" -> force cuEquivariance
            anything else          -> "auto" (try cueq first, fall back to e3nn)
        """
        if not isinstance(tp_type, str):
            return "auto"
        tags = {token.strip().lower() for token in tp_type.split("+") if token.strip()}
        if "tp_e3nn" in tags or "e3nn_tp" in tags:
            return "e3nn"
        if "tp_cueq" in tags or "cueq_tp" in tags:
            return "cueq"
        return "auto"

    @staticmethod
    def _build_initial_radial_coeffs(
        *,
        lmax: int,
        num_heads: int,
        kappa: torch.Tensor,
        mode: str,
        init_scale: float,
        low_kappa_bias: float,
    ) -> tuple[torch.Tensor, bool]:
        """Build initial spectral-mixture coefficients with low-kappa preference.

        Modes:
            - "per_l_head":   shape (L, H, Q), one radial profile per (l, head)
            - "per_l_shared": shape (L, Q),    one radial profile per l
            - "head":         shape (H, Q),    shared over l, per-head
            - "shared":       shape (Q,),      shared over l and heads
        """
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"per_l_head", "per_l_shared", "head", "shared"}:
            raise ValueError(
                "fmm_radial_coeffs_mode must be one of "
                "{'per_l_head','per_l_shared','head','shared'}, "
                f"got {mode!r}"
            )

        # Normalize kappa into [0,1], then apply an exponential decay so lower
        # frequencies get larger initial weights than high-frequency modes.
        denom = (kappa.max() - kappa.min()).clamp_min(1e-6)
        kappa_01 = (kappa - kappa.min()) / denom
        low_kappa_profile = torch.exp(-float(low_kappa_bias) * kappa_01)
        low_kappa_profile = low_kappa_profile / low_kappa_profile.sum().clamp_min(1e-8)
        base = float(init_scale) * low_kappa_profile

        num_l = int(lmax) + 1
        if mode_norm == "per_l_head":
            return base[None, None, :].repeat(num_l, int(num_heads), 1), True
        if mode_norm == "per_l_shared":
            return base[None, :].repeat(num_l, 1), True
        if mode_norm == "head":
            return base[None, :].repeat(int(num_heads), 1), False
        return base.clone(), False

    @staticmethod
    def _build_tp_right_from_cueq(
        *,
        irreps_ell: o3.Irreps,
        irreps_lambda: o3.Irreps,
    ) -> torch.Tensor | None:
        """Attempt to build the CG right-projection matrix using cuEquivariance.

        cuEquivariance provides GPU-accelerated Clebsch-Gordan tensor products.
        This method constructs the same right-projection matrix R[Q, P, R_total]
        as e3nn's FullTensorProduct.right(), but using cuEquivariance's kernels.

        Strategy: feed an identity basis through the cuEquivariance TP to extract
        the linear map as an explicit matrix. We create Q*P input pairs where
        each pair activates exactly one (ell_component, lambda_component), then
        read off the output to build R.

        Returns:
            right_all: (Q, P, R_total) tensor on CPU if successful, else None.
            Returns None if cuEquivariance is not installed or fails.
        """
        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except Exception:
            return None

        use_cuda = torch.cuda.is_available() and _cueq_ops_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        use_fallback = not use_cuda

        try:
            cue_irreps_ell = cue.Irreps("O3", str(irreps_ell))
            cue_irreps_lambda = cue.Irreps("O3", str(irreps_lambda))
            descriptor = cue.descriptors.full_tensor_product(
                cue_irreps_ell,
                cue_irreps_lambda,
            )
            tp_module = cuet.EquivariantTensorProduct(
                descriptor,
                layout=cue.mul_ir,
                device=device,
                use_fallback=use_fallback,
            )
        except Exception as err:
            warnings.warn(
                f"Failed to initialize cuequivariance TP backend, falling back to e3nn ({err})."
            )
            return None

        # Build identity-basis inputs to probe the TP:
        # x1_basis: (Q*P, P) -- cycles through all P ell-components Q times
        # x2_basis: (Q*P, Q) -- one-hot for the corresponding lambda-component
        q_dim = int(irreps_lambda.dim)  # Q = (lmax+1)^2
        p_dim = int(irreps_ell.dim)     # P = (lmax+1)^2
        x1_basis = torch.eye(p_dim, device=device, dtype=torch.float32).repeat(q_dim, 1)
        q_indices = torch.arange(q_dim, device=device, dtype=torch.long).repeat_interleave(p_dim)
        x2_basis = torch.zeros((q_dim * p_dim, q_dim), device=device, dtype=torch.float32)
        x2_basis.scatter_(1, q_indices.unsqueeze(1), 1.0)
        try:
            right_flat = tp_module(x1_basis, x2_basis)  # (Q*P, R_total)
        except Exception as err:
            warnings.warn(
                f"Failed to run cuequivariance TP backend, falling back to e3nn ({err})."
            )
            return None

        # Reshape from (Q*P, R_total) -> (Q, P, R_total) for consistency with e3nn.
        right_all = right_flat.view(q_dim, p_dim, -1).to(device="cpu", dtype=torch.float32)
        return right_all.contiguous()

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """Run the full node-only FMM attention layer.

        This method is the main entry point, designed to be a drop-in replacement
        for the original E2Former attention module. It accepts the same arguments
        (many of which are discarded since this module does not use edges).

        High-level steps:
          1. Pack flat node tensors into padded per-graph batches.
          2. Project irreps to scalar Q/K per head; reshape irreps as multi-degree V.
          3. Run the FMM core to get per-ell spherical expansion of attention*V.
          4. Apply precomputed CG coupling to combine ell (FMM) and lambda (value)
             degrees into output degrees L.
          5. Average coupling paths, unpack to flat node indexing, return.

        Args:
            node_pos:          (N, 3) node positions (flat across all graphs).
            node_irreps_input: (N, (lmax+1)^2, C) SO(3) node features.
            edge_dis:          Unused (no edges in this module).
            edge_vec:          Unused.
            attn_weight:       Passed through unchanged (for interface compatibility).
            atomic_numbers:    Unused.
            poly_dist:         Unused.
            attn_mask:         Unused.
            batch:             (N,) graph-membership vector. If None, inferred from
                               batched_data["f_batch"] or assumed single graph.
            batched_data:      Optional dict; may contain "f_batch" as fallback.

        Returns:
            (node_output, attn_weight) where:
                node_output: (N, (lmax+1)^2, C) equivariant output features.
                attn_weight: passed through unchanged from input.
        """
        # Discard unused edge/atom arguments (this module is fully node-based).
        del edge_dis, edge_vec, atomic_numbers, poly_dist, attn_mask, kwargs

        # Early exit for empty inputs (e.g., empty batch).
        if node_irreps_input.numel() == 0:
            return node_irreps_input, attn_weight

        # =====================================================================
        # Step 1: Resolve the batch (graph-membership) vector
        # =====================================================================
        # The batch vector maps each node to its graph index. Three fallbacks:
        #   1. Explicit `batch` argument
        #   2. batched_data["f_batch"] (E2Former convention for fractional coords)
        #   3. All nodes belong to a single graph (batch = zeros)
        if batch is None:
            if batched_data is not None and "f_batch" in batched_data:
                batch = batched_data["f_batch"]
            else:
                batch = torch.zeros(
                    node_pos.shape[0], dtype=torch.long, device=node_pos.device
                )
        batch = batch.to(device=node_pos.device, dtype=torch.long)

        # =====================================================================
        # Step 2: Pack nodes into padded per-graph batches
        # =====================================================================
        # Convert from flat (N, ...) to padded (B, max_nodes, ...) layout.
        # The FMM core requires regular tensor shapes for batched computation.
        packed_pos, packed_irreps, packed_mask, pack_info = self._pack_by_graph(
            node_pos=node_pos,
            node_irreps=node_irreps_input,
            batch=batch,
        )
        num_graphs, max_nodes = packed_pos.shape[:2]

        # =====================================================================
        # Step 3: Q/K projections (irreps -> scalar per head)
        # =====================================================================
        # Project in fp32 for numerical stability (SO3_Linear2Scalar involves
        # degree-wise inner products that are sensitive to precision).
        # packed_irreps: (B, max_nodes, (lmax+1)^2, C) -> q,k: (B, max_nodes, H, d)
        packed_irreps_f = packed_irreps.to(dtype=torch.float32)
        q = self.q_proj(packed_irreps_f).view(
            num_graphs, max_nodes, self.num_attn_heads, self.attn_scalar_head
        )
        k = self.k_proj(packed_irreps_f).view(
            num_graphs, max_nodes, self.num_attn_heads, self.attn_scalar_head
        )

        # =====================================================================
        # Step 4: Prepare value irreps for FMM
        # =====================================================================
        # Allocate the final output tensor (same shape as input irreps).
        value_dim_total = self.num_attn_heads * self.value_head_dim
        packed_out = packed_irreps.new_zeros(
            (num_graphs, max_nodes, (self.lmax + 1) ** 2, value_dim_total)
        )
        num_irrep_components = self.num_irrep_components  # = (lmax+1)^2

        # Reshape the node irreps into per-head value tensors.
        # The input has shape (B, max_nodes, (lmax+1)^2, C) where C = H * Cv.
        # We split C into (H, Cv) and move the head axis:
        #   (B, max_nodes, Q, H, Cv) -> permute -> (B, max_nodes, H, Q, Cv)
        # This keeps all lambda-components together per head for the FMM.
        if self.v_proj is None:
            packed_v = packed_irreps
        else:
            # (B, N, Q, C) -> (B, N, Q, H*Cv_small)
            packed_v = self.v_proj(packed_irreps)

        v_all = packed_v.view(
            num_graphs,
            max_nodes,
            num_irrep_components,   # Q = (lmax+1)^2 lambda components
            self.num_attn_heads,    # H
            self.value_head_dim,    # Cv (possibly bottlenecked)
        ).permute(0, 1, 3, 2, 4).contiguous()  # (B, max_nodes, H, Q, Cv)

        # Flatten the (Q, Cv) dimensions into a single value dimension for the
        # FMM core, which expects v of shape (B, max_nodes, H, V_dim).
        # After FMM, each output degree ell will carry this flattened dim,
        # which we'll unpack back to (Q, Cv) for CG coupling.
        v_all_flat = v_all.view(
            num_graphs,
            max_nodes,
            self.num_attn_heads,
            num_irrep_components * self.value_head_dim,  # Q * Cv
        )

        # =====================================================================
        # Step 5: Run the FMM core (all degrees simultaneously)
        # =====================================================================
        # The FMM core computes, for each degree ell = 0..lmax:
        #   out_ell[i, h, m_ell, V_dim] =
        #       (1/Z_i) * sum_{q,s} a_q * phi(q_i)^T M_{q,s} * psi_{q,s}(r_i)
        #                          * W_ell[s, m_ell]
        # where the V_dim carries the flattened (lambda, Cv) value information.
        #
        # Returns a list of tensors, one per ell:
        #   out_by_ell[ell] has shape (B, max_nodes, H, (2*ell+1) * Q * Cv)
        #   (from the FMM core's perspective, the last dim is just "value features")
        fmm_pos = packed_pos
        if self.fmm_compute_dtype is not None and isinstance(self.fmm_compute_dtype, str):
            token = self.fmm_compute_dtype.strip().lower()
            if token not in ("", "auto", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "half"):
                raise ValueError(
                    "fmm_compute_dtype must be one of {'auto','fp32','bf16','fp16'} "
                    f"(or synonymous strings), got {self.fmm_compute_dtype!r}."
                )
            if token in ("bf16", "bfloat16"):
                fmm_pos = fmm_pos.to(dtype=torch.bfloat16)
            elif token in ("fp16", "float16", "half"):
                fmm_pos = fmm_pos.to(dtype=torch.float16)
            elif token in ("fp32", "float32"):
                fmm_pos = fmm_pos.to(dtype=torch.float32)

        out_by_ell = self.fmm_multi_l(
            fmm_pos,
            q,
            k,
            v_all_flat,
            node_mask=packed_mask,
        )

        # =====================================================================
        # Step 6: CG coupling -- combine FMM degree (ell) with value degree (lambda)
        # =====================================================================
        # Unpack all ell blocks and concatenate the m_ell axis into a single P axis:
        #   out_by_ell[ell]: (B, N, H, (2*ell+1)*Q*Cv)
        #   ->              (B, N, H, 2*ell+1, Q, Cv)
        #   cat over ell -> (B, N, H, P, Q, Cv), P=(lmax+1)^2
        out_blocks: list[torch.Tensor] = []
        for ell, out_ell_flat in enumerate(out_by_ell):
            m_ell = 2 * ell + 1
            out_blocks.append(
                out_ell_flat.view(
                    num_graphs,
                    max_nodes,
                    self.num_attn_heads,
                    m_ell,
                    num_irrep_components,
                    self.value_head_dim,
                )
            )

        if not out_blocks:
            raise RuntimeError("FMM coupling produced no output blocks.")
        out_all = torch.cat(out_blocks, dim=3)  # (B, N, H, P, Q, Cv)
        if out_all.shape[3] != self.num_ell_components:
            raise RuntimeError(
                "Unexpected fused ell dimension in TP coupling: "
                f"{out_all.shape[3]} vs expected {self.num_ell_components}."
            )

        # Single fused CG coupling GEMM across all (ell,m_ell,lambda):
        #   (B, N, H, P, Q, Cv) x (P*Q, R_total) -> (B, N, H, R_total, Cv)
        tp_right_flat_all = self.tp_right_flat_all
        if (
            tp_right_flat_all.device != out_all.device
            or tp_right_flat_all.dtype != out_all.dtype
        ):
            tp_right_flat_all = tp_right_flat_all.to(
                device=out_all.device,
                dtype=out_all.dtype,
            )
        out_all_flat = out_all.permute(0, 1, 2, 5, 3, 4).reshape(
            -1,
            self.num_ell_components * num_irrep_components,
        )
        coupled_all = out_all_flat.matmul(tp_right_flat_all).view(
            num_graphs,
            max_nodes,
            self.num_attn_heads,
            self.value_head_dim,
            self.tp_out_dim,
        ).permute(0, 1, 2, 4, 3).contiguous()  # (B, N, H, R_total, Cv)

        # =====================================================================
        # Step 7: Extract output degree-L blocks from the coupled TP output
        # =====================================================================
        # The TP output irreps contain blocks for all degrees L that can arise
        # from coupling ell x lambda. We extract only L <= lmax blocks and
        # place them into the corresponding slice of the packed output.
        #
        # For each (out_l, start, end, mul) in tp_out_blocks:
        #   - out_l: the output degree L
        #   - start:end: slice into the R_total dimension of coupled_all
        #   - mul: number of (ell, lambda) paths producing this L (may be > 1)
        #
        # When mul > 1, the multiple paths are summed (averaged later).
        for out_l, start, end, mul in self.tp_out_blocks:
            m_out = 2 * out_l + 1
            block = coupled_all[:, :, :, start:end, :]  # (B, N, H, mul*m_out, Cv)
            if mul > 1:
                # Multiple coupling paths -> split and sum them.
                block = block.view(
                    num_graphs,
                    max_nodes,
                    self.num_attn_heads,
                    mul,
                    m_out,
                    self.value_head_dim,
                ).sum(dim=3)  # (B, N, H, m_out, Cv)
            else:
                block = block.view(
                    num_graphs,
                    max_nodes,
                    self.num_attn_heads,
                    m_out,
                    self.value_head_dim,
                )

            # Rearrange from (B, N, H, m_out, Cv) to (B, N, m_out, H*Cv) = (B, N, m_out, C)
            # to match the input irreps layout.
            block = block.permute(0, 1, 3, 2, 4).contiguous()
            block = block.reshape(num_graphs, max_nodes, m_out, value_dim_total)

            # Place into the correct slice of the packed output.
            out_start, out_end = self._l_slice(out_l)
            packed_out[:, :, out_start:out_end, :].add_(block)

        # =====================================================================
        # Step 8: Normalize by the number of coupling paths
        # =====================================================================
        # Each output degree L receives contributions from coupling_count[L]
        # different (ell, lambda) paths. We average (not just sum) to keep the
        # output scale independent of how many CG paths exist for each L.
        for out_l in range(self.lmax + 1):
            out_start, out_end = self._l_slice(out_l)
            packed_out[:, :, out_start:out_end, :].div_(self.coupling_count[out_l])

        if self.out_proj is not None:
            packed_out = self.out_proj(packed_out)

        # =====================================================================
        # Step 9: Unpack back to flat node indexing and return
        # =====================================================================
        # Convert from (B, max_nodes, (lmax+1)^2, C) back to (N, (lmax+1)^2, C).
        node_output = self._unpack_by_graph(
            packed_output=packed_out,
            pack_info=pack_info,
            n_nodes=node_irreps_input.shape[0],
        )
        # Return (output, attn_weight) to match the original E2Former attention interface.
        # attn_weight is passed through unchanged (this module does not produce explicit
        # attention weight matrices since it uses linear attention internally).
        return node_output, attn_weight
