# -*- coding: utf-8 -*-
"""Prototype implementations of the alphaf(r)Y equivariant attention operator.

This module provides three attention-kernel variants and a high-level wrapper:

Classes:
    AlphaFRYSphericalSoftmax:
        O(N^2) exact baseline using standard softmax attention weights.
        Computes:  out_i = _j  softmax_j(q_ik_j / sqrt(d))   f(|r_ij|)  Y_l(r_ij)  v_j

    AlphaFRYSphericalFMM:
        O(NSQ) separable (FMM-style) approximation using *linear* attention.
        The plane-wave expansion  j_l(r)Y_l(r)  i^{-l} _s w_s Y_l(u_s) e^{i u_sr_i} e^{-i u_sr_j}
        decouples source and target nodes, making the contraction over j factorizable.
        Combined with linear attention (  <(q), (k)>), the entire operation avoids
        explicitly forming the NN attention/geometry matrices.

    E2FormerAlphaFRYPrototype:
        High-level wrapper that builds Q/K/V projections from E2Former-style irreps
        and dispatches to one of the above core operators.

Mathematical background:
    The core operation computed by all variants is:
        out[i, h, m, c] = _j  [i,h,j]  f_l(|r_ij|)  Y_{l,m}(r_ij)  v[j,h,c]

    where:
        - [i,h,j] are per-head attention weights (softmax or linear)
        - f_l(r) = _q a_q  j_l(_q  r)  is a radial function modeled as a mixture
          of spherical Bessel functions of the first kind
        - Y_{l,m}(r) are real spherical harmonics of degree l, order m
        - v[j,h,c] are per-head value features at source node j

    The result has shape (*shape, H, 2l+1, Cv):
        - H heads, each producing (2l+1) spherical-harmonic components of dimension Cv.

    For l=0 this reduces to a standard (radially-modulated) attention; for l>0 the
    output lives in a higher-order irreducible representation of SO(3), enabling the
    network to produce equivariant vector/tensor features.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .fmm_exact import AlphaFRYSphericalLinear, AlphaFRYSphericalVanilla
from .fmm_utils import (
    infer_compute_dtype,
    positive_feature_map,
    prepare_radial_coeffs,
    prod,
    spherical_bessel_j,
    spherical_harmonics_real,
)
from .sphere_quadrature import (
    e3nn_s2grid_sphere,
    fibonacci_sphere,
    gauss_legendre_sphere,
    lebedev_sphere,
)

# ---------------------------
# Shape / symbol conventions
# ---------------------------
# B: batch size
# num_nodes: number of nodes per graph (sequence length)
# N: total nodes after flattening leading dims, N = prod(*shape) (e.g. N=B*num_nodes)
# H: number of attention heads
# d: per-head query/key feature dimension
# Cv: per-head value feature dimension (value_dim)
# M: number of m-components for spherical harmonics at degree l, M = 2*l + 1
# Q: number of radial frequencies (kappa/a mixture size)
# S: number of sphere quadrature directions (num_directions)
#
# Typical tensor layouts used in this file:
# - pos: (*shape, 3)  e.g. (B, num_nodes, 3)
# - q,k: (*shape, H, d)
# - v:   (*shape, H, Cv)
# - output core (headed): (*shape, H, M, Cv)
# - output core (head-merged): (*shape, M, H*Cv) or (*shape, M, Cv) if H=1
#
# 
#                   E2FormerAlphaFRYPrototype.forward                   
#                                                                      
#   Inputs: node_pos (B,num_nodes,3), node_irreps (B,num_nodes,(lmax+1),C)          
# 
#                             
#             
#                                           
#      
#       Q Projection  K Projection    V Projection   
#     SO3_Linear2     SO3_Linear2    nn.Linear on    
#     Scalar(irreps)  Scalar(irr.)   scalar (l=0)    
#      (B,num_nodes,H,d)     (B,num_nodes,H,d)    (B,num_nodes,H,Cv)   
#      
#                                              
#            
#                             
# 
#                AlphaFRYSphericalFMM.forward (core)                   
#                                                                      
#   Inputs: pos (B,num_nodes,3), q (B,num_nodes,H,d), k (B,num_nodes,H,d), v (B,num_nodes,H,Cv)    
# 
#                             
#               
#                                          
#        
#      Flatten to (N,...)        Positive Feature Map 
#      N = B*num_nodes                   (q), (k) via      
#      pos_f: (N,3)              ELU(x)+1             
#        
#                                          
#                              
#                                                     
#                      
#                    Denominator Z        Apply node_mask    
#                    s = _j (k_j)      (zero out padding) 
#                    Z_ih = <(q_i),    
#                            s_h>     
#                   
#                            
#                            
#     
#     Precompute geometry   
#                           
#     W[s,m] = w_sY_l(u_s)    quadrature weights  spherical harmonics
#     dot[n,s] = r_nu_s       position-direction dot products
#     
#                            
#                            
#   
#            Loop over radial frequencies q = 1..Q             
#                                                              
#       
#       phase[n,s] = _q  dot[n,s]                         
#       [n,s] = exp(i  phase) = cos(phase) + isin(ph.)   
#       
#                                                             
#                                                             
#       
#       STEP 1: Source Moment Formation (aggregate j)        
#                                                            
#       M[s,h,d,c] = _n (k_n)[h,d]  *(n,s)  v[n,h,c] 
#                                                            
#       (This is the "multipole expansion"  precomputes     
#        source contributions independently of targets)      
#       
#                                                             
#                                                             
#       
#       STEP 2: Query Projection (contract feature dim d)    
#                                                            
#       b[n,s,h,c] = _d (q_n)[h,d]  M[s,h,d,c]         
#                                                            
#       (Target-side attention weighting)                    
#       
#                                                             
#                                                             
#       
#       STEP 3: Local Evaluation                             
#                                                            
#       term[n,s,h,c] = (n,s)  b[n,s,h,c]                
#                                                            
#       (Reapply plane-wave phase at target position)        
#       
#                                                             
#                                                             
#       
#       STEP 4: Sphere Projection                            
#                                                            
#       out_q[n,h,m,c] = _s term[n,s,h,c]  W[s,m]        
#                                                            
#       (Contract quadrature directions into Y_lm basis)     
#       
#                                                             
#                                                             
#       
#       Accumulate: out += a_q    out_q                   
#       ( = i^{-l} phase factor, a_q = radial coefficient)  
#       
#   
#                           
#                           
#           
#             Normalize: out / Z[n,h]      
#             (linear-attention denominator)
#             Take real part               
#           
#                           
#                           
# 
#               Back in E2FormerAlphaFRYPrototype.forward               
#                                                                      
#   Reshape: (N, H, M, Cv)  (B, num_nodes, H, M, Cv)                        
#   Permute:  (B, num_nodes, M, H, Cv)                                       
#   Merge heads:  (B, num_nodes, M, H*Cv)                                    
#   Optionally pack into full irreps embedding (B, num_nodes, (lmax+1), C')  
# 

class AlphaFRYSphericalSoftmax(nn.Module):
    """Exact alphaf(r)Y operator with *softmax* attention weights (O(N^2)).

    This is the closest "standard attention" baseline:
        alpha[i,h,j] = softmax_j( (q[i,h]k[j,h]) / sqrt(d) )

    The full computation is:
        out[i, h, m, c] = _j  alpha[i,h,j]  f_l(|r_ij|)  Y_{l,m}(r_ij)  v[j,h,c]

    This forms the complete NN attention matrix and NN pairwise geometry tensors
    explicitly, so it scales as O(N^2) in both time and memory. It serves as the
    gold-standard reference for validating the cheaper FMM approximation.

    Args:
        l:              Spherical harmonic degree for the angular kernel.
                        l=0 gives scalar output, l=1 gives vector (3-component), etc.
        kappa:          1D tensor of shape (Q,)  radial frequencies for the Bessel basis.
        a:              Radial mixing coefficients. Supported shapes:
                          - (Q,)      shared across all heads and value channels
                          - (H, Q)    per-head coefficients
                          - (H,C,Q)   per-head-per-channel coefficients
        normalization:  Spherical harmonic normalization convention (e3nn convention).
        sh_backend:     Spherical-harmonics backend ("auto" | "e3nn" | "cueq").
        eps:            Small constant to avoid division by zero in softmax denominator.
        scale:          Optional manual scaling of qk scores (default: 1/sqrt(d)).
    """

    def __init__(
        self,
        l: int,
        kappa: torch.Tensor,
        a: torch.Tensor,
        *,
        normalization: str = "integral",
        sh_backend: str = "auto",
        eps: float = 1e-8,
        scale: Optional[float] = None,
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
        self.normalization = normalization
        self.sh_backend = sh_backend
        self.eps = float(eps)
        self.scale = scale

        # Keep radial basis parameters as *buffers* (not Parameters) so they:
        #   1. Move with the module on .to(device), .to(dtype), etc.
        #   2. Are included in state_dict for checkpointing.
        #   3. Are NOT updated by the optimizer (they are fixed basis functions).
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
        """Compute the softmax-attention alphaf(r)Y operator.

        This is the O(N^2) reference implementation that:
          1. Explicitly forms all pairwise attention weights _{i,j} via softmax.
          2. Computes the full NN pairwise distance matrix and spherical harmonics.
          3. Contracts over source nodes j to produce equivariant output.

        The more scalable FMM path (AlphaFRYSphericalFMM) avoids the explicit NN
        matrices by using the plane-wave / spherical quadrature factorization.

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
        # Perform geometry-heavy computation in fp32 for numerical stability under AMP,
        # since spherical harmonics and Bessel functions are sensitive to precision.
        pos_f = pos.reshape(B, num_nodes, 3).to(dtype=compute_dtype)

        q_flat = q.reshape((B, num_nodes) + q.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        k_flat = k.reshape((B, num_nodes) + k.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        v_flat = v.reshape((B, num_nodes) + v.shape[pos.ndim - 1 :])

        # Track whether the caller passed headed tensors so we can restore the
        # same layout on output (squeeze H=1 if the caller didn't use heads).
        had_heads = q_flat.ndim == 4

        # Support both headed (B,num_nodes,H,d) and unheaded (B,num_nodes,d) inputs.
        # If unheaded, insert a singleton head dimension: (B,num_nodes,d) -> (B,num_nodes,1,d).
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
            # Allow v to be broadcast across heads if it is provided without heads.
            # This is common when values are shared but queries/keys are per-head.
            v_flat = v_flat.expand(-1, -1, H, -1)
        elif v_flat.shape[2] != H:
            raise ValueError(f"v must have head dim 1 or H={H}, got {v_flat.shape[2]}")
        C = v_flat.shape[-1]

        # =====================================================================
        # Phase 2: Masking setup
        # =====================================================================
        # node_mask handles variable-length graphs within a padded batch.
        # Mask semantics: True/1 = valid node; False/0 = padding node.
        key_mask_f: Optional[torch.Tensor] = None
        query_mask_f: Optional[torch.Tensor] = None
        if node_mask is not None:
            if node_mask.shape != lead_shape:
                raise ValueError(f"node_mask must be {tuple(lead_shape)}, got {tuple(node_mask.shape)}")
            m = node_mask.reshape(B, num_nodes).to(device=pos.device)
            # Key mask: prevents attending *to* invalid (padding) keys.
            # Query mask: zeros out outputs *from* invalid (padding) queries.
            key_mask_f = m.to(dtype=compute_dtype)  # (B, num_nodes)
            query_mask_f = key_mask_f
            # Also zero out values at padding positions so they don't leak signal.
            v_flat = v_flat * key_mask_f[:, :, None, None].to(dtype=v_flat.dtype)

        v_compute = v_flat.to(dtype=compute_dtype)

        # =====================================================================
        # Phase 3: Softmax attention weights  [b,h,i,j]
        # =====================================================================
        d = q_flat.shape[-1]
        scale = float(self.scale) if self.scale is not None else 1.0 / math.sqrt(float(d))

        # Compute raw attention scores per graph:
        # scores[b,h,i,j] = (q[b,i,h]  k[b,j,h]) / sqrt(d).
        # Einsum index map:
        #   q_flat(b,i,h,d)  k_flat(b,j,h,d) -> scores(b,h,i,j)
        #   reduce: d (inner product)
        #   keep: b (graph), h (head), i (query/target), j (key/source)
        scores = torch.einsum("bihd,bjhd->bhij", q_flat, k_flat) * scale  # (B, H, num_nodes, num_nodes)
        if key_mask_f is not None:
            # Use a very negative value at padding keys so exp(score)  0 in softmax.
            scores = scores.masked_fill(key_mask_f[:, None, None, :] == 0, -1.0e9)

        # Numerically stable softmax: subtract max before exp to prevent overflow.
        scores = scores - scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores)
        if key_mask_f is not None:
            # Belt-and-suspenders: explicitly zero masked keys after exp too.
            exp_scores = exp_scores * key_mask_f[:, None, None, :]
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        alpha = exp_scores / denom  # (B, H, num_nodes, num_nodes)  normalized attention weights
        if query_mask_f is not None:
            # Zero out entire rows for padding query nodes.
            alpha = alpha * query_mask_f[:, None, :, None]

        # =====================================================================
        # Phase 4: Pairwise geometry  displacements, distances, SH, Bessel
        # =====================================================================

        # Build all pairwise displacement vectors within each graph:
        # r[b,i,j] = pos[b,i] - pos[b,j].
        r_ij = pos_f[:, :, None, :] - pos_f[:, None, :, :]  # (B, num_nodes, num_nodes, 3)
        dist = torch.linalg.vector_norm(r_ij, dim=-1)  # (B, num_nodes, num_nodes)

        # The diagonal (i==j) has r_ij = 0, which causes NaNs in
        # spherical_harmonics(normalize=True) because it divides by ||r||.
        # We set a safe nonzero direction on the diagonal; the actual
        # f(0)Y(r) product is well-defined and correct because j_l(0)=0
        # for l>0 (and the direction is irrelevant for l=0).
        r_ij_safe = r_ij.clone()
        diag = torch.arange(num_nodes, device=pos.device)
        r_ij_safe[:, diag, diag] = torch.tensor(
            [1.0, 0.0, 0.0], device=pos.device, dtype=compute_dtype
        )

        # Angular factor: Y_{l,m}(r_ij)  real spherical harmonics evaluated at
        # the unit direction of each displacement vector.
        Y = spherical_harmonics_real(
            self.l,
            r_ij_safe,
            normalize=True,
            normalization=self.normalization,
            backend=self.sh_backend,
        )  # (B, num_nodes, num_nodes, 2l+1)

        # Radial factor: f_l(r) = _q  a_q  j_l(_q  r)
        # The radial function is expanded in a basis of spherical Bessel functions
        # of the first kind.  This is a natural basis for expansions on [0, )
        # that arises from the Helmholtz equation in spherical coordinates.
        x = dist[..., None] * self.kappa.to(dist.dtype)  # (B, num_nodes, num_nodes, Q)  _q * r_ij
        jl = spherical_bessel_j(self.l, x, eps=self.eps)  # (B, num_nodes, num_nodes, Q)  j_l(_q r_ij)

        # Prepare radial coefficients in the appropriate broadcasting mode.
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
        # The final contraction computes:
        #   out[i,h,m,c] = _j  [h,i,j]  f(r_ij)  Y_{l,m}(r_ij)  v[j,h,c]
        #
        # The form of f depends on a_mode (scalar, per-head, or per-head-channel):

        if a_mode == "scalar":
            # f(r_ij) is the same for all heads and channels.
            f = torch.einsum("bijq,q->bij", jl, a_coeff)  # (B, num_nodes, num_nodes)
            # Combined weight: attention  radial
            w = alpha * f[:, None, :, :]  # (B, H, num_nodes, num_nodes)
            # Contract over source nodes j:
            #   w(b,h,i,j)  Y(b,i,j,r)  v(b,j,h,c) -> out(b,i,h,r,c)
            out = torch.einsum("bhij,bijr,bjhc->bihrc", w, Y, v_compute)
        elif a_mode == "head":
            # f depends on head h: different radial profile per attention head.
            f = torch.einsum("bijq,hq->bhij", jl, a_coeff)  # (B, H, num_nodes, num_nodes)
            w = alpha * f
            out = torch.einsum("bhij,bijr,bjhc->bihrc", w, Y, v_compute)
        else:
            # f depends on both head h and value channel c (most expressive).
            f = torch.einsum("bijq,hcq->bhijc", jl, a_coeff)  # (B, H, num_nodes, num_nodes, C')
            if f.shape[-1] == 1 and C > 1:
                f = f.expand(-1, -1, -1, -1, C)  # broadcast singleton channel
            w = alpha.unsqueeze(-1) * f  # (B, H, num_nodes, num_nodes, C)
            out = torch.einsum("bhijc,bijr,bjhc->bihrc", w, Y, v_compute)

        # =====================================================================
        # Phase 6: Restore original shape and return
        # =====================================================================
        out = out.to(dtype=v.dtype)
        out = out.reshape(lead_shape + (H, 2 * self.l + 1, v_flat.shape[-1]))
        if not had_heads and q.ndim == pos.ndim and v.ndim == pos.ndim:
            out = out.squeeze(-3)  # (*shape, 2l+1, c)  remove singleton head dim
        return out


class AlphaFRYSphericalFMM(nn.Module):
    """Separable (FMM-style) approximation for the alphaf(r)Y operator using linear attention.

    Mathematical background
    -----------------------
    The exact operator requires an O(N^2) sum:
        out_i = _j  _{ij}  j_l(|r_ij|)  Y_l(r_ij)  v_j

    The FMM approximation replaces the pairwise kernel with a **plane-wave expansion**:
        j_l(|r_ij|)  Y_l(r_ij)    i^{-l}  _s  w_s  Y_l(u_s)  e^{i u_sr_i}  e^{-i u_sr_j}

    The key observation is that the right-hand side is *separable* in i and j:
    the factor e^{i u_sr_i} depends only on the target, and e^{-i u_sr_j}
    depends only on the source. This means we can first aggregate all source
    contributions into "moments" (a sum over j), then evaluate at each target i,
    reducing O(N^2) to O(NS) per radial frequency.

    Combined with **linear attention** (  <(q), (k)>), the attention weights
    are also separable, so the entire operation avoids any NN matrix.

    This corresponds to the node-wise moment factorization in `FMM/fmm.tex`
    (Theorem "True node-wise factorization of equivariant attention"):
      - key sum:     s = _j (k_j)
      - moments:     M_{q,s} = sum_j phi(k_j) * (psi_{q,s}(r_j)^* v_j)
      - normalization: Z_i = <(q_i), s>
    and evaluates outputs from node-local query contractions with those moments.

    Complexity: O(N  S  Q  H  d) time, O(N  S + S  H  d  C) memory,
    where S = num_directions, Q = num radial frequencies  all independent of N^2.

    Sphere quadrature options
    -------------------------
    The quality of the approximation depends on the sphere rule {u_s, w_s}:
      - "lebedev":        High-accuracy polynomial-exact quadrature (recommended).
      - "fibonacci":      Equal-weight quasi-uniform lattice (simple, no SciPy needed).
      - "gauss_legendre": Tensor-product rule (Gauss-Legendre  uniform ).
      - "e3nn_s2grid":    Band-limited Kostelec-Rockmore grid from e3nn.

    Args:
        l:               Spherical harmonic degree (l=0: scalar, l=1: vector, ...).
        kappa:           Radial frequencies, shape (Q,).
        a:               Radial mixing coefficients. Shape (Q,), (H,Q), or (H,C,Q).
        num_directions:  Number of quadrature points S on the sphere.
        sphere:          Quadrature scheme name (see above).
        num_theta/num_phi:     Optional overrides for gauss_legendre factorization.
        res_beta/res_alpha:    Optional overrides for e3nn_s2grid grid resolution.
        phase_mode:      "complex" uses complex arithmetic (reference);
                         "trig" uses cos/sin real channels (GPU-friendly, equivalent).
        kappa_chunk_size:
                         Number of radial frequencies processed together in the
                         per- loop. Values >1 reduce launch overhead at the
                         cost of higher temporary-memory usage.
                         Set to 0 to enable auto mode:
                           - start from chunk=Q (max speed),
                           - if CUDA OOM occurs, retry with chunk halved.
        max_auto_kappa_chunk:
                         Upper bound for the initial auto-chunk trial when
                         `kappa_chunk_size=0`. Helps avoid very large temporary
                         tensors in high-dimensional settings.
        allow_tf32:      Enable TF32 tensor-core math for CUDA float32 matmuls.
                         This typically speeds up FMM significantly on Ampere/Hopper.
        matmul_precision:
                         Passed to torch.set_float32_matmul_precision when TF32 is
                         enabled ("high" is usually the best speed/accuracy tradeoff).
        promote_half_precision:
                         If True (default), promote fp16/bf16 inputs to fp32 for
                         geometry-heavy contractions (more stable).
                         If False, keep native fp16/bf16 compute for maximum
                         throughput (faster but less numerically stable).
        optimize_low_precision_sphere:
                         When True, apply a low-precision quadrature heuristic in
                         the fast trig path:
                           - if using low-precision native compute
                             (`promote_half_precision=False`),
                           - and `sphere='lebedev'` with small `num_directions`,
                         switch to a Gauss-Legendre grid that tends to reduce
                         approximation error at nearly identical runtime.
        center_positions:
                         If True, subtract a per-graph position center before
                         phase evaluation. This preserves relative geometry while
                         improving finite-precision stability for large absolute
                         coordinates (especially in mixed precision).
        sh_backend:      Spherical-harmonics backend ("auto" | "e3nn" | "cueq").
        feature_map:     Positive feature map for linear attention ("elu" or "softplus").
        normalization:   e3nn spherical harmonic normalization convention.
        eps:             Small constant for numerical stability.
    """

    def __init__(
        self,
        l: int,
        kappa: torch.Tensor,
        a: torch.Tensor,
        *,
        num_directions: int = 64,
        sphere: str = "lebedev",  # "lebedev" | "fibonacci" | ("e3nn_s2grid" / "gauss_legendre" experimental)
        num_theta: Optional[int] = None,
        num_phi: Optional[int] = None,
        res_beta: Optional[int] = None,
        res_alpha: Optional[int] = None,
        phase_mode: str = "complex",  # "complex" (reference) | "trig" (cos/sin channels)
        kappa_chunk_size: int = 0,
        max_auto_kappa_chunk: int = 40,
        allow_tf32: bool = True,
        matmul_precision: str = "high",
        promote_half_precision: bool = True,
        optimize_low_precision_sphere: bool = True,
        center_positions: bool = True,
        sh_backend: str = "auto",
        feature_map: str = "elu",
        normalization: str = "integral",
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
        if num_directions <= 0:
            raise ValueError(f"num_directions must be > 0, got {num_directions}")
        if kappa_chunk_size < 0:
            raise ValueError(f"kappa_chunk_size must be >= 0, got {kappa_chunk_size}")
        if max_auto_kappa_chunk < 0:
            raise ValueError(
                f"max_auto_kappa_chunk must be >= 0, got {max_auto_kappa_chunk}"
            )
        if matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(
                "matmul_precision must be one of {'highest','high','medium'}, "
                f"got {matmul_precision!r}"
            )
        if sh_backend not in ("auto", "e3nn", "cueq"):
            raise ValueError(
                "sh_backend must be one of {'auto','e3nn','cueq'}, "
                f"got {sh_backend!r}"
            )

        self.l = int(l)
        self.feature_map = feature_map
        self.normalization = normalization
        self.eps = float(eps)
        self.kappa_chunk_size = int(kappa_chunk_size)
        self.max_auto_kappa_chunk = int(max_auto_kappa_chunk)
        self.allow_tf32 = bool(allow_tf32)
        self.matmul_precision = matmul_precision
        self.promote_half_precision = bool(promote_half_precision)
        self.optimize_low_precision_sphere = bool(optimize_low_precision_sphere)
        self.center_positions = bool(center_positions)
        self.sh_backend = sh_backend
        self.last_kappa_chunk: Optional[int] = None

        # Register radial basis parameters as buffers (see softmax class for rationale).
        self.register_buffer("kappa", kappa.detach().clone())
        self.register_buffer("a", a.detach().clone())

        if sphere not in ("lebedev", "fibonacci", "e3nn_s2grid", "gauss_legendre"):
            raise ValueError(
                "sphere must be 'lebedev', 'fibonacci', 'e3nn_s2grid', or 'gauss_legendre', "
                f"got {sphere!r}"
            )

        if phase_mode not in ("trig", "complex"):
            raise ValueError(f"phase_mode must be 'trig' or 'complex', got {phase_mode!r}")
        self.phase_mode = phase_mode

        # Low-precision fast-path heuristic:
        # For very small direction counts, a Gauss-Legendre product grid can be
        # more accurate than small Lebedev rules while keeping similar throughput.
        if (
            self.optimize_low_precision_sphere
            and (not self.promote_half_precision)
            and self.phase_mode == "trig"
            and sphere == "lebedev"
            and int(num_directions) <= 26
            and num_theta is None
            and num_phi is None
        ):
            sphere = "gauss_legendre"
            # Keep point count close to 26 while using a well-factored product grid.
            # 25 = 5x5 works well in practice for the fast low-precision path.
            num_directions = 25
        self.sphere = sphere

        # =====================================================================
        # Build sphere quadrature {u_s, w_s} once at init time on CPU.
        # These are moved to the correct device in forward() via .to(device).
        # This keeps __init__ cheap and avoids GPU allocation during model setup.
        # =====================================================================
        cpu = torch.device("cpu")
        if sphere == "lebedev":
            # Lebedev-Laikov quadrature: high-quality rule that integrates spherical
            # polynomials of degree  order exactly. Best accuracy per point, but
            # only available at specific point counts (6, 14, 26, 38, 50, 74, ...).
            dirs, w, self.lebedev_order, self.lebedev_points = lebedev_sphere(
                int(num_directions),
                device=cpu,
                dtype=torch.float32,
            )
        elif sphere == "fibonacci":
            # Fibonacci lattice: simple quasi-uniform distribution with equal weights.
            # Equivalent to Monte-Carlo integration with deterministic points.
            # No external dependencies; good for quick prototyping.
            dirs = fibonacci_sphere(num_directions, device=cpu, dtype=torch.float32)
            w = torch.full((num_directions,), 1.0 / float(num_directions), dtype=torch.float32)
        elif sphere == "gauss_legendre":
            # Tensor-product quadrature: Gauss-Legendre nodes in cos() combined
            # with uniform trapezoid rule in . Exact for tensor-product integrands.
            dirs, w, self.num_theta, self.num_phi = gauss_legendre_sphere(
                num_directions,
                num_theta=num_theta,
                num_phi=num_phi,
                device=cpu,
                dtype=torch.float32,
            )
        else:
            # e3nn's S2 grid: band-limited Kostelec-Rockmore grid used internally
            # by e3nn for spherical harmonic transforms. Spectral accuracy.
            dirs, w, self.res_beta, self.res_alpha = e3nn_s2grid_sphere(
                num_directions,
                l_required=self.l,
                res_beta=res_beta,
                res_alpha=res_alpha,
                device=cpu,
                dtype=torch.float32,
            )

        # Store directions and weights as buffers so they survive .to() / state_dict.
        # These define the discrete sphere integral:
        #   _{S^2} g(u) du    _s  w_s  g(u_s)
        self.register_buffer("directions", dirs)  # (S, 3)  unit vectors on S^2
        self.register_buffer("sphere_weights", w)  # (S,)   quadrature weights, sum=1

        # Precompute Y_l(u_s) for all quadrature directions once.
        # These only depend on the sphere rule and degree l, so caching them avoids
        # recomputing spherical harmonics every forward pass.
        Y_dirs = spherical_harmonics_real(
            self.l,
            dirs,
            normalize=True,
            normalization=self.normalization,
            backend=self.sh_backend,
        )  # (S, 2l+1)
        self.register_buffer("Y_dirs", Y_dirs)

        # The phase factor c_l = i^{-l} that appears in the plane-wave expansion.
        # For integer l:  i^{-l} cycles through {1, -i, -1, i} as l mod 4 = {0,1,2,3}.
        # We store l mod 4 to efficiently apply this factor without complex arithmetic
        # in the "trig" path.
        self._l_mod_4 = int(self.l % 4)

    def forward(
        self,
        pos: torch.Tensor,  # (*shape, 3)
        q: torch.Tensor,  # (*shape, d) or (*shape, H, d)
        k: torch.Tensor,  # (*shape, d) or (*shape, H, d)
        v: torch.Tensor,  # (*shape, c) or (*shape, H, c)
        node_mask: Optional[torch.Tensor] = None,  # (*shape,)
    ) -> torch.Tensor:
        """Compute the FMM-approximated linear-attention alphaf(r)Y operator.

        High-level algorithm (for each radial frequency _q):
          1. Compute plane-wave phases: [n,s] = exp(i_q  u_s  r_n)
          2. Source moment formation: aggregate (k), *, and v over all nodes n
             into a compact moment tensor M[s,h,d,c].  (FMM "multipole" step)
          3. Query projection: contract (q) with M to get per-target-per-direction
             coefficients b[n,s,h,c].
          4. Local evaluation: reapply  at the target position.
          5. Sphere projection: contract over quadrature directions s with
             W[s,m] = w_sY_l(u_s) to produce the (2l+1) spherical-harmonic components.
          6. Accumulate over radial frequencies q with mixing coefficients a_q.
        Finally, divide by the linear-attention denominator Z.

        The computation matches the edge-free node-wise form:
          out_i ~ (1/Z_i) * sum_{q,s} CG(phi(q_i)^T M_{q,s}, Y(u_s)) * psi_{q,s}(r_i)
        implemented in real/complex equivalent arithmetic paths.

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
        # Keep graph boundaries explicit: treat the last leading axis as the node
        # axis num_nodes and flatten only the preceding axes into a graph axis B.
        # This prevents cross-graph leakage when inputs are batched (B>1).
        graph_shape = lead_shape[:-1]
        B = int(prod(graph_shape)) if len(graph_shape) > 0 else 1
        num_nodes = int(lead_shape[-1])

        compute_dtype = (
            infer_compute_dtype(pos.dtype) if self.promote_half_precision else pos.dtype
        )
        pos_f = pos.reshape(B, num_nodes, 3).to(dtype=compute_dtype)

        q_flat = q.reshape((B, num_nodes) + q.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        k_flat = k.reshape((B, num_nodes) + k.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        v_flat = v.reshape((B, num_nodes) + v.shape[pos.ndim - 1 :])

        # Track whether caller used explicit heads, for correct output shape.
        had_heads = q_flat.ndim == 4
        if q_flat.ndim == 3:
            q_flat = q_flat.unsqueeze(2)  # (B, num_nodes, 1, d)  treat unheaded as H=1
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
            v_flat = v_flat.expand(-1, -1, H, -1)
        elif v_flat.shape[2] != H:
            raise ValueError(f"v must have head dim 1 or H={H}, got {v_flat.shape[2]}")

        # =====================================================================
        # Phase 2: Positive feature maps and masking (linear attention setup)
        # =====================================================================
        # Linear attention replaces softmax with a kernel:
        #   alpha(i,j) = <phi(q_i), phi(k_j)>
        # The feature map phi must be non-negative so that the denominator
        #   Z_i = sum_j <phi(q_i), phi(k_j)> = <phi(q_i), sum_j phi(k_j)>
        # can be computed in O(N) instead of O(N^2).
        m: Optional[torch.Tensor] = None
        if node_mask is not None:
            if node_mask.shape != lead_shape:
                raise ValueError(f"node_mask must be {tuple(lead_shape)}, got {tuple(node_mask.shape)}")
            m = node_mask.reshape(B, num_nodes).to(device=pos.device, dtype=compute_dtype)

        if self.center_positions:
            # Center each graph independently to reduce phase-drift from large
            # absolute coordinates. With masks, use only valid nodes.
            if m is None:
                center = pos_f.mean(dim=1, keepdim=True)  # (B, 1, 3)
                pos_f = pos_f - center
            else:
                denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
                center = (pos_f * m[:, :, None]).sum(dim=1, keepdim=True) / denom[:, :, None]
                pos_f = (pos_f - center) * m[:, :, None]

        phi_q = positive_feature_map(q_flat, kind=self.feature_map)  # (B, num_nodes, H, d)
        phi_k = positive_feature_map(k_flat, kind=self.feature_map)  # (B, num_nodes, H, d)

        if m is not None:
            # Zero out features at padding positions so padded nodes contribute
            # neither to moments nor to normalization.
            phi_q = phi_q * m[:, :, None, None]
            phi_k = phi_k * m[:, :, None, None]
            v_flat = v_flat * m[:, :, None, None].to(dtype=v_flat.dtype)

        # =====================================================================
        # Phase 3: Linear-attention denominator (per-graph O(BLHd))
        # =====================================================================
        # Keep the graph axis explicit: all reductions are over node axis num_nodes only.
        key_sum = phi_k.sum(dim=1)  # (B, H, d)
        Z = torch.einsum("blhd,bhd->blh", phi_q, key_sum).clamp_min(self.eps)  # (B, num_nodes, H)

        # =====================================================================
        # Phase 4: Precompute shared geometry for the plane-wave expansion
        # =====================================================================
        dirs = self.directions.to(device=pos.device, dtype=compute_dtype)  # (S, 3)
        w = self.sphere_weights.to(device=pos.device, dtype=compute_dtype)  # (S,)
        Y_dirs = self.Y_dirs.to(device=pos.device, dtype=compute_dtype)  # (S, 2l+1)

        W = w[:, None] * Y_dirs  # (S, 2l+1)
        dot = torch.einsum("blc,sc->bls", pos_f, dirs)  # (B, num_nodes, S)

        M = 2 * self.l + 1
        C = v_flat.shape[-1]

        kappa = self.kappa.to(device=pos.device, dtype=compute_dtype)  # (Q,)
        a_coeff, a_mode = prepare_radial_coeffs(
            self.a,
            num_kappa=int(kappa.shape[0]),
            num_heads=H,
            value_dim=C,
            device=pos.device,
            dtype=compute_dtype,
        )
        v_compute = v_flat.to(dtype=compute_dtype)

        restore_tf32 = False
        prev_allow_tf32 = False
        prev_matmul_precision = "highest"
        is_compiling = hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()
        if (
            self.allow_tf32
            and pos.device.type == "cuda"
            and compute_dtype == torch.float32
            and not is_compiling
        ):
            restore_tf32 = True
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_matmul_precision = torch.get_float32_matmul_precision()
            torch.backends.cuda.matmul.allow_tf32 = True
            if prev_matmul_precision != self.matmul_precision:
                torch.set_float32_matmul_precision(self.matmul_precision)

        # =====================================================================
        # Phase 5: Main FMM loop over radial frequencies
        # =====================================================================
        try:
            if self.kappa_chunk_size == 0:
                if self.last_kappa_chunk is not None:
                    q_chunk_start = min(int(self.last_kappa_chunk), int(kappa.shape[0]))
                else:
                    q_chunk_start = int(kappa.shape[0])
            else:
                q_chunk_start = min(int(self.kappa_chunk_size), int(kappa.shape[0]))
            auto_chunk = self.kappa_chunk_size == 0 and pos.device.type == "cuda"
            if auto_chunk and self.max_auto_kappa_chunk > 0:
                q_chunk_start = min(q_chunk_start, self.max_auto_kappa_chunk)
            q_chunk = max(1, int(q_chunk_start))

            while True:
                try:
                    if self.phase_mode == "complex":
                        complex_dtype = (
                            torch.complex64
                            if compute_dtype in (torch.float16, torch.bfloat16, torch.float32)
                            else torch.complex128
                        )
                        W_c = W.to(dtype=complex_dtype)
                        out_c = torch.zeros((B, num_nodes, H, M, C), device=pos.device, dtype=complex_dtype)
                        phi_k_c = phi_k.to(dtype=complex_dtype)
                        phi_q_c = phi_q.to(dtype=complex_dtype)
                        v_c = v_compute.to(dtype=complex_dtype)

                        c_l = complex((1j) ** (-self.l))
                        c_l_t = torch.tensor(c_l, device=pos.device, dtype=complex_dtype)
                        gamma = a_coeff.to(dtype=complex_dtype) * c_l_t

                        for q_start in range(0, int(kappa.shape[0]), q_chunk):
                            q_end = min(q_start + q_chunk, int(kappa.shape[0]))
                            kappa_chunk = kappa[q_start:q_end]  # (T,)
                            phase = dot[:, :, :, None] * kappa_chunk[None, None, None, :]  # (B, num_nodes, S, T)
                            psi = torch.cos(phase).to(complex_dtype) + 1j * torch.sin(phase).to(complex_dtype)

                            # Sum over node axis l only, per graph b.
                            # Fused form avoids materializing kv_c[b,l,h,d,c].
                            moment_sthdc = torch.einsum(
                                "blst,blhd,blhc->bsthdc",
                                psi.conj(),
                                phi_k_c,
                                v_c,
                            )
                            b_nsthc = torch.einsum("blhd,bsthdc->blsthc", phi_q_c, moment_sthdc)

                            t_chunk = int(q_end - q_start)
                            use_fused_proj = t_chunk >= 8
                            if a_mode == "scalar":
                                gamma_chunk = gamma[q_start:q_end]
                                if use_fused_proj:
                                    out_c.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,t->blhmc",
                                            psi,
                                            b_nsthc,
                                            W_c,
                                            gamma_chunk,
                                        )
                                    )
                                else:
                                    out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                    out_c.add_(torch.einsum("blthmc,t->blhmc", out_chunk, gamma_chunk))
                            elif a_mode == "head":
                                gamma_chunk = gamma[:, q_start:q_end]
                                if use_fused_proj:
                                    out_c.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,ht->blhmc",
                                            psi,
                                            b_nsthc,
                                            W_c,
                                            gamma_chunk,
                                        )
                                    )
                                else:
                                    out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                    out_c.add_(torch.einsum("blthmc,ht->blhmc", out_chunk, gamma_chunk))
                            else:
                                out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                gamma_chunk = gamma[:, :, q_start:q_end]  # (H, C', T)
                                if gamma_chunk.shape[1] == 1 and C > 1:
                                    gamma_chunk = gamma_chunk.expand(-1, C, -1)
                                out_c.add_(torch.einsum("blthmc,hct->blhmc", out_chunk, gamma_chunk))

                        out = (
                            out_c / Z[:, :, :, None, None].to(dtype=complex_dtype)
                        ).real.to(dtype=v.dtype)

                    else:
                        out_real = torch.zeros((B, num_nodes, H, M, C), device=pos.device, dtype=compute_dtype)
                        out_imag = torch.zeros((B, num_nodes, H, M, C), device=pos.device, dtype=compute_dtype)
                        for q_start in range(0, int(kappa.shape[0]), q_chunk):
                            q_end = min(q_start + q_chunk, int(kappa.shape[0]))
                            kappa_chunk = kappa[q_start:q_end]  # (T,)

                            phase = dot[:, :, :, None] * kappa_chunk[None, None, None, :]  # (B, num_nodes, S, T)
                            cos_p = torch.cos(phase)
                            sin_p = torch.sin(phase)

                            # Fuse phi_k * v into the contraction to avoid a large
                            # temporary kv tensor with shape (B, num_nodes, H, d, C).
                            moment_real = torch.einsum("blst,blhd,blhc->bsthdc", cos_p, phi_k, v_compute)
                            moment_imag = -torch.einsum("blst,blhd,blhc->bsthdc", sin_p, phi_k, v_compute)

                            b_real = torch.einsum("blhd,bsthdc->blsthc", phi_q, moment_real)
                            b_imag = torch.einsum("blhd,bsthdc->blsthc", phi_q, moment_imag)

                            t_chunk = int(q_end - q_start)
                            use_fused_proj = t_chunk >= 8
                            if a_mode == "scalar":
                                scale_chunk = a_coeff[q_start:q_end]
                                if use_fused_proj:
                                    out_real.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,t->blhmc",
                                            cos_p,
                                            b_real,
                                            W,
                                            scale_chunk,
                                        )
                                        - torch.einsum(
                                            "blst,blsthc,sm,t->blhmc",
                                            sin_p,
                                            b_imag,
                                            W,
                                            scale_chunk,
                                        )
                                    )
                                    out_imag.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,t->blhmc",
                                            cos_p,
                                            b_imag,
                                            W,
                                            scale_chunk,
                                        )
                                        + torch.einsum(
                                            "blst,blsthc,sm,t->blhmc",
                                            sin_p,
                                            b_real,
                                            W,
                                            scale_chunk,
                                        )
                                    )
                                else:
                                    out_real_chunk = torch.einsum("blst,blsthc,sm->blthmc", cos_p, b_real, W) - torch.einsum(
                                        "blst,blsthc,sm->blthmc",
                                        sin_p,
                                        b_imag,
                                        W,
                                    )
                                    out_imag_chunk = torch.einsum("blst,blsthc,sm->blthmc", cos_p, b_imag, W) + torch.einsum(
                                        "blst,blsthc,sm->blthmc",
                                        sin_p,
                                        b_real,
                                        W,
                                    )
                                    out_real.add_(torch.einsum("blthmc,t->blhmc", out_real_chunk, scale_chunk))
                                    out_imag.add_(torch.einsum("blthmc,t->blhmc", out_imag_chunk, scale_chunk))
                            elif a_mode == "head":
                                scale_chunk = a_coeff[:, q_start:q_end]  # (H, T)
                                if use_fused_proj:
                                    out_real.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,ht->blhmc",
                                            cos_p,
                                            b_real,
                                            W,
                                            scale_chunk,
                                        )
                                        - torch.einsum(
                                            "blst,blsthc,sm,ht->blhmc",
                                            sin_p,
                                            b_imag,
                                            W,
                                            scale_chunk,
                                        )
                                    )
                                    out_imag.add_(
                                        torch.einsum(
                                            "blst,blsthc,sm,ht->blhmc",
                                            cos_p,
                                            b_imag,
                                            W,
                                            scale_chunk,
                                        )
                                        + torch.einsum(
                                            "blst,blsthc,sm,ht->blhmc",
                                            sin_p,
                                            b_real,
                                            W,
                                            scale_chunk,
                                        )
                                    )
                                else:
                                    out_real_chunk = torch.einsum("blst,blsthc,sm->blthmc", cos_p, b_real, W) - torch.einsum(
                                        "blst,blsthc,sm->blthmc",
                                        sin_p,
                                        b_imag,
                                        W,
                                    )
                                    out_imag_chunk = torch.einsum("blst,blsthc,sm->blthmc", cos_p, b_imag, W) + torch.einsum(
                                        "blst,blsthc,sm->blthmc",
                                        sin_p,
                                        b_real,
                                        W,
                                    )
                                    out_real.add_(torch.einsum("blthmc,ht->blhmc", out_real_chunk, scale_chunk))
                                    out_imag.add_(torch.einsum("blthmc,ht->blhmc", out_imag_chunk, scale_chunk))
                            else:
                                out_real_chunk = torch.einsum(
                                    "blst,blsthc,sm->blthmc",
                                    cos_p,
                                    b_real,
                                    W,
                                ) - torch.einsum(
                                    "blst,blsthc,sm->blthmc",
                                    sin_p,
                                    b_imag,
                                    W,
                                )
                                out_imag_chunk = torch.einsum(
                                    "blst,blsthc,sm->blthmc",
                                    cos_p,
                                    b_imag,
                                    W,
                                ) + torch.einsum(
                                    "blst,blsthc,sm->blthmc",
                                    sin_p,
                                    b_real,
                                    W,
                                )
                                scale_chunk = a_coeff[:, :, q_start:q_end]  # (H, C', T)
                                if scale_chunk.shape[1] == 1 and C > 1:
                                    scale_chunk = scale_chunk.expand(-1, C, -1)
                                out_real.add_(torch.einsum("blthmc,hct->blhmc", out_real_chunk, scale_chunk))
                                out_imag.add_(torch.einsum("blthmc,hct->blhmc", out_imag_chunk, scale_chunk))

                        if self._l_mod_4 == 0:
                            out_acc = out_real
                        elif self._l_mod_4 == 1:
                            out_acc = out_imag
                        elif self._l_mod_4 == 2:
                            out_acc = -out_real
                        else:
                            out_acc = -out_imag

                        out = (out_acc / Z[:, :, :, None, None]).to(dtype=v.dtype)
                    self.last_kappa_chunk = int(q_chunk)
                    break
                except RuntimeError as exc:
                    oom = "out of memory" in str(exc).lower()
                    if not (auto_chunk and oom and q_chunk > 1):
                        raise
                    if pos.device.type == "cuda":
                        torch.cuda.empty_cache()
                    q_chunk = max(1, q_chunk // 2)
        finally:
            if restore_tf32:
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                if prev_matmul_precision != self.matmul_precision:
                    torch.set_float32_matmul_precision(prev_matmul_precision)

        # =====================================================================
        # Phase 6: Restore original shape and return
        # =====================================================================
        out = out.reshape(lead_shape + (H, M, C))
        if not had_heads and q.ndim == pos.ndim and v.ndim == pos.ndim:
            out = out.squeeze(-3)  # remove singleton head dim
        return out


class AlphaFRYSphericalFMMMultiL(nn.Module):
    """Multi-degree FMM core that shares geometry/moment work across multiple l values.

    This class computes the same operation as running multiple `AlphaFRYSphericalFMM`
    instances (one per l), but shares the heavy steps that are independent of l:
      - feature-map transforms and normalization denominator,
      - position/direction dot products and phase tensors,
      - source moment construction and query projection.

    Only the final sphere projection `W_l = w * Y_l(u)` and the `i^{-l}` selection differ
    per l. This significantly reduces duplicated work in node-FMM blocks that need all
    degrees `l = 0..lmax`.

    When `kappa_chunk_size=0` (auto mode), `max_auto_kappa_chunk` bounds the initial
    chunk size to control temporary-memory pressure on large `(B,num_nodes,H,C,Q,S)` workloads.
    Spherical harmonics can be evaluated via e3nn or cuEquivariance (when available).
    """

    def __init__(
        self,
        l_values: list[int],
        kappa: torch.Tensor,
        a: torch.Tensor,
        *,
        num_directions: int = 64,
        sphere: str = "lebedev",
        num_theta: Optional[int] = None,
        num_phi: Optional[int] = None,
        res_beta: Optional[int] = None,
        res_alpha: Optional[int] = None,
        phase_mode: str = "complex",
        kappa_chunk_size: int = 0,
        max_auto_kappa_chunk: int = 40,
        allow_tf32: bool = True,
        matmul_precision: str = "high",
        promote_half_precision: bool = True,
        optimize_low_precision_sphere: bool = True,
        center_positions: bool = True,
        sh_backend: str = "auto",
        feature_map: str = "elu",
        normalization: str = "integral",
        eps: float = 1e-8,
        a_per_l: bool = False,
        learnable_a: bool = False,
    ) -> None:
        super().__init__()
        if len(l_values) == 0:
            raise ValueError("l_values must be non-empty")
        if any(l < 0 for l in l_values):
            raise ValueError(f"all l_values must be >= 0, got {l_values}")
        if len(set(l_values)) != len(l_values):
            raise ValueError(f"l_values must be unique, got {l_values}")
        self.l_values = sorted(int(l) for l in l_values)
        self._l_to_index = {int(l): idx for idx, l in enumerate(self.l_values)}
        self.lmax = max(self.l_values)

        if kappa.ndim != 1:
            raise ValueError(f"kappa must be 1D of shape (Q,), got {tuple(kappa.shape)}")
        self.a_per_l = bool(a_per_l)
        if self.a_per_l:
            if a.ndim not in (2, 3, 4):
                raise ValueError(
                    "When a_per_l=True, a must have shape (L,Q), (L,H,Q), or (L,H,C,Q), "
                    f"got {tuple(a.shape)}"
                )
            if a.shape[0] != len(self.l_values):
                raise ValueError(
                    f"When a_per_l=True, a.shape[0] must equal number of l values={len(self.l_values)}, "
                    f"got {a.shape[0]}"
                )
        else:
            if a.ndim not in (1, 2, 3):
                raise ValueError(
                    f"a must have shape (Q,), (H,Q), or (H,C,Q), got {tuple(a.shape)}"
                )
        if a.shape[-1] != kappa.shape[0]:
            raise ValueError(
                f"a.shape[-1] must equal Q={kappa.shape[0]}, got a.shape={tuple(a.shape)}"
            )
        if num_directions <= 0:
            raise ValueError(f"num_directions must be > 0, got {num_directions}")
        if kappa_chunk_size < 0:
            raise ValueError(f"kappa_chunk_size must be >= 0, got {kappa_chunk_size}")
        if max_auto_kappa_chunk < 0:
            raise ValueError(
                f"max_auto_kappa_chunk must be >= 0, got {max_auto_kappa_chunk}"
            )
        if matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(
                "matmul_precision must be one of {'highest','high','medium'}, "
                f"got {matmul_precision!r}"
            )
        if phase_mode not in ("trig", "complex"):
            raise ValueError(f"phase_mode must be 'trig' or 'complex', got {phase_mode!r}")
        if sh_backend not in ("auto", "e3nn", "cueq"):
            raise ValueError(
                "sh_backend must be one of {'auto','e3nn','cueq'}, "
                f"got {sh_backend!r}"
            )

        self.feature_map = feature_map
        self.normalization = normalization
        self.eps = float(eps)
        self.phase_mode = phase_mode
        self.kappa_chunk_size = int(kappa_chunk_size)
        self.max_auto_kappa_chunk = int(max_auto_kappa_chunk)
        self.allow_tf32 = bool(allow_tf32)
        self.matmul_precision = matmul_precision
        self.promote_half_precision = bool(promote_half_precision)
        self.optimize_low_precision_sphere = bool(optimize_low_precision_sphere)
        self.center_positions = bool(center_positions)
        self.sh_backend = sh_backend
        self.last_kappa_chunk: Optional[int] = None
        self.learnable_a = bool(learnable_a)

        self.register_buffer("kappa", kappa.detach().clone())
        if self.learnable_a:
            self.register_parameter("a", nn.Parameter(a.detach().clone()))
        else:
            self.register_buffer("a", a.detach().clone())

        if sphere not in ("lebedev", "fibonacci", "e3nn_s2grid", "gauss_legendre"):
            raise ValueError(
                "sphere must be 'lebedev', 'fibonacci', 'e3nn_s2grid', or 'gauss_legendre', "
                f"got {sphere!r}"
            )
        if (
            self.optimize_low_precision_sphere
            and (not self.promote_half_precision)
            and self.phase_mode == "trig"
            and sphere == "lebedev"
            and int(num_directions) <= 26
            and num_theta is None
            and num_phi is None
        ):
            sphere = "gauss_legendre"
            num_directions = 25
        self.sphere = sphere

        cpu = torch.device("cpu")
        if sphere == "lebedev":
            dirs, w, self.lebedev_order, self.lebedev_points = lebedev_sphere(
                int(num_directions),
                device=cpu,
                dtype=torch.float32,
            )
        elif sphere == "fibonacci":
            dirs = fibonacci_sphere(num_directions, device=cpu, dtype=torch.float32)
            w = torch.full((num_directions,), 1.0 / float(num_directions), dtype=torch.float32)
        elif sphere == "gauss_legendre":
            dirs, w, self.num_theta, self.num_phi = gauss_legendre_sphere(
                num_directions,
                num_theta=num_theta,
                num_phi=num_phi,
                device=cpu,
                dtype=torch.float32,
            )
        else:
            dirs, w, self.res_beta, self.res_alpha = e3nn_s2grid_sphere(
                num_directions,
                l_required=self.lmax,
                res_beta=res_beta,
                res_alpha=res_alpha,
                device=cpu,
                dtype=torch.float32,
            )

        self.register_buffer("directions", dirs)
        self.register_buffer("sphere_weights", w)

        for l in self.l_values:
            Y_dirs = spherical_harmonics_real(
                l,
                dirs,
                normalize=True,
                normalization=self.normalization,
                backend=self.sh_backend,
            )
            self.register_buffer(f"Y_dirs_l{l}", Y_dirs)
        self._l_mod_4 = [int(l % 4) for l in self.l_values]

    def forward(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if pos.ndim < 2 or pos.shape[-1] != 3:
            raise ValueError(f"pos must be (*shape, 3), got {tuple(pos.shape)}")
        if q.shape[: pos.ndim - 1] != pos.shape[:-1] or k.shape[: pos.ndim - 1] != pos.shape[:-1]:
            raise ValueError("q and k must share the same leading *shape as pos")
        if v.shape[: pos.ndim - 1] != pos.shape[:-1]:
            raise ValueError("v must share the same leading *shape as pos")

        lead_shape = pos.shape[:-1]
        graph_shape = lead_shape[:-1]
        B = int(prod(graph_shape)) if len(graph_shape) > 0 else 1
        num_nodes = int(lead_shape[-1])

        compute_dtype = (
            infer_compute_dtype(pos.dtype) if self.promote_half_precision else pos.dtype
        )
        pos_f = pos.reshape(B, num_nodes, 3).to(dtype=compute_dtype)
        q_flat = q.reshape((B, num_nodes) + q.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        k_flat = k.reshape((B, num_nodes) + k.shape[pos.ndim - 1 :]).to(dtype=compute_dtype)
        v_flat = v.reshape((B, num_nodes) + v.shape[pos.ndim - 1 :])

        had_heads = q_flat.ndim == 4
        if q_flat.ndim == 3:
            q_flat = q_flat.unsqueeze(2)
        if k_flat.ndim == 3:
            k_flat = k_flat.unsqueeze(2)
        if q_flat.ndim != 4 or k_flat.ndim != 4:
            raise ValueError("q and k must be (*shape,d) or (*shape,H,d)")
        if q_flat.shape[2:] != k_flat.shape[2:]:
            raise ValueError(
                f"q and k must match (H,d), got {q_flat.shape[2:]} vs {k_flat.shape[2:]}"
            )

        if v_flat.ndim == 3:
            v_flat = v_flat.unsqueeze(2)
        if v_flat.ndim != 4:
            raise ValueError("v must be (*shape,c) or (*shape,H,c)")

        H = q_flat.shape[2]
        if v_flat.shape[2] == 1 and H > 1:
            v_flat = v_flat.expand(-1, -1, H, -1)
        elif v_flat.shape[2] != H:
            raise ValueError(f"v must have head dim 1 or H={H}, got {v_flat.shape[2]}")

        m: Optional[torch.Tensor] = None
        if node_mask is not None:
            if node_mask.shape != lead_shape:
                raise ValueError(f"node_mask must be {tuple(lead_shape)}, got {tuple(node_mask.shape)}")
            m = node_mask.reshape(B, num_nodes).to(device=pos.device, dtype=compute_dtype)

        if self.center_positions:
            if m is None:
                center = pos_f.mean(dim=1, keepdim=True)
                pos_f = pos_f - center
            else:
                denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                center = (pos_f * m[:, :, None]).sum(dim=1, keepdim=True) / denom[:, :, None]
                pos_f = (pos_f - center) * m[:, :, None]

        phi_q = positive_feature_map(q_flat, kind=self.feature_map)
        phi_k = positive_feature_map(k_flat, kind=self.feature_map)
        if m is not None:
            phi_q = phi_q * m[:, :, None, None]
            phi_k = phi_k * m[:, :, None, None]
            v_flat = v_flat * m[:, :, None, None].to(dtype=v_flat.dtype)

        key_sum = phi_k.sum(dim=1)
        Z = torch.einsum("blhd,bhd->blh", phi_q, key_sum).clamp_min(self.eps)

        dirs = self.directions.to(device=pos.device, dtype=compute_dtype)
        w = self.sphere_weights.to(device=pos.device, dtype=compute_dtype)
        W_list = [
            (w[:, None] * getattr(self, f"Y_dirs_l{l}").to(device=pos.device, dtype=compute_dtype))
            for l in self.l_values
        ]

        dot = torch.einsum("blc,sc->bls", pos_f, dirs)
        C = v_flat.shape[-1]
        kappa = self.kappa.to(device=pos.device, dtype=compute_dtype)
        a_coeff_list: list[torch.Tensor] = []
        if self.a_per_l:
            if int(self.a.shape[0]) != len(self.l_values):
                raise RuntimeError(
                    f"a_per_l expects first dim L={len(self.l_values)}, got {int(self.a.shape[0])}"
                )
            a_mode: Optional[str] = None
            for l_idx in range(len(self.l_values)):
                a_coeff_l, a_mode_l = prepare_radial_coeffs(
                    self.a[l_idx],
                    num_kappa=int(kappa.shape[0]),
                    num_heads=H,
                    value_dim=C,
                    device=pos.device,
                    dtype=compute_dtype,
                )
                if a_mode is None:
                    a_mode = a_mode_l
                elif a_mode_l != a_mode:
                    raise RuntimeError(
                        "All per-l radial coefficients must share the same broadcast mode, "
                        f"got {a_mode!r} and {a_mode_l!r}."
                    )
                a_coeff_list.append(a_coeff_l)
            if a_mode is None:
                raise RuntimeError("Failed to infer radial coefficient mode for per-l coefficients.")
        else:
            a_coeff_shared, a_mode = prepare_radial_coeffs(
                self.a,
                num_kappa=int(kappa.shape[0]),
                num_heads=H,
                value_dim=C,
                device=pos.device,
                dtype=compute_dtype,
            )
            a_coeff_list = [a_coeff_shared] * len(self.l_values)
        v_compute = v_flat.to(dtype=compute_dtype)

        restore_tf32 = False
        prev_allow_tf32 = False
        prev_matmul_precision = "highest"
        is_compiling = hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()
        if (
            self.allow_tf32
            and pos.device.type == "cuda"
            and compute_dtype == torch.float32
            and not is_compiling
        ):
            restore_tf32 = True
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_matmul_precision = torch.get_float32_matmul_precision()
            torch.backends.cuda.matmul.allow_tf32 = True
            if prev_matmul_precision != self.matmul_precision:
                torch.set_float32_matmul_precision(self.matmul_precision)

        try:
            if self.kappa_chunk_size == 0:
                if self.last_kappa_chunk is not None:
                    q_chunk_start = min(int(self.last_kappa_chunk), int(kappa.shape[0]))
                else:
                    q_chunk_start = int(kappa.shape[0])
            else:
                q_chunk_start = min(int(self.kappa_chunk_size), int(kappa.shape[0]))
            auto_chunk = self.kappa_chunk_size == 0 and pos.device.type == "cuda"
            if auto_chunk and self.max_auto_kappa_chunk > 0:
                q_chunk_start = min(q_chunk_start, self.max_auto_kappa_chunk)
            q_chunk = max(1, int(q_chunk_start))

            while True:
                try:
                    if self.phase_mode == "complex":
                        complex_dtype = (
                            torch.complex64
                            if compute_dtype in (torch.float16, torch.bfloat16, torch.float32)
                            else torch.complex128
                        )
                        W_c_list = [W.to(dtype=complex_dtype) for W in W_list]
                        out_c_list = [
                            torch.zeros(
                                (B, num_nodes, H, 2 * l + 1, C), device=pos.device, dtype=complex_dtype
                            )
                            for l in self.l_values
                        ]
                        phi_k_c = phi_k.to(dtype=complex_dtype)
                        phi_q_c = phi_q.to(dtype=complex_dtype)
                        v_c = v_compute.to(dtype=complex_dtype)
                        gamma_list = []
                        for idx, l in enumerate(self.l_values):
                            c_l = complex((1j) ** (-l))
                            c_l_t = torch.tensor(c_l, device=pos.device, dtype=complex_dtype)
                            gamma_list.append(a_coeff_list[idx].to(dtype=complex_dtype) * c_l_t)

                        for q_start in range(0, int(kappa.shape[0]), q_chunk):
                            q_end = min(q_start + q_chunk, int(kappa.shape[0]))
                            kappa_chunk = kappa[q_start:q_end]
                            phase = dot[:, :, :, None] * kappa_chunk[None, None, None, :]
                            psi = torch.cos(phase).to(complex_dtype) + 1j * torch.sin(phase).to(complex_dtype)
                            # Fused form avoids materializing kv_c[b,l,h,d,c].
                            moment_sthdc = torch.einsum(
                                "blst,blhd,blhc->bsthdc",
                                psi.conj(),
                                phi_k_c,
                                v_c,
                            )
                            b_nsthc = torch.einsum("blhd,bsthdc->blsthc", phi_q_c, moment_sthdc)

                            t_chunk = int(q_end - q_start)
                            use_fused_proj = t_chunk >= 8
                            for idx, gamma in enumerate(gamma_list):
                                W_c = W_c_list[idx]
                                out_c = out_c_list[idx]
                                if a_mode == "scalar":
                                    gamma_chunk = gamma[q_start:q_end]
                                    if use_fused_proj:
                                        out_c.add_(
                                            torch.einsum(
                                                "blst,blsthc,sm,t->blhmc",
                                                psi,
                                                b_nsthc,
                                                W_c,
                                                gamma_chunk,
                                            )
                                        )
                                    else:
                                        out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                        out_c.add_(torch.einsum("blthmc,t->blhmc", out_chunk, gamma_chunk))
                                elif a_mode == "head":
                                    gamma_chunk = gamma[:, q_start:q_end]
                                    if use_fused_proj:
                                        out_c.add_(
                                            torch.einsum(
                                                "blst,blsthc,sm,ht->blhmc",
                                                psi,
                                                b_nsthc,
                                                W_c,
                                                gamma_chunk,
                                            )
                                        )
                                    else:
                                        out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                        out_c.add_(torch.einsum("blthmc,ht->blhmc", out_chunk, gamma_chunk))
                                else:
                                    out_chunk = torch.einsum("blst,blsthc,sm->blthmc", psi, b_nsthc, W_c)
                                    gamma_chunk = gamma[:, :, q_start:q_end]
                                    if gamma_chunk.shape[1] == 1 and C > 1:
                                        gamma_chunk = gamma_chunk.expand(-1, C, -1)
                                    out_c.add_(torch.einsum("blthmc,hct->blhmc", out_chunk, gamma_chunk))

                        out_list = [
                            (out_c / Z[:, :, :, None, None].to(dtype=complex_dtype)).real.to(dtype=v.dtype)
                            for out_c in out_c_list
                        ]
                    else:
                        out_real_list = [
                            torch.zeros((B, num_nodes, H, 2 * l + 1, C), device=pos.device, dtype=compute_dtype)
                            for l in self.l_values
                        ]
                        out_imag_list = [
                            torch.zeros((B, num_nodes, H, 2 * l + 1, C), device=pos.device, dtype=compute_dtype)
                            for l in self.l_values
                        ]
                        for q_start in range(0, int(kappa.shape[0]), q_chunk):
                            q_end = min(q_start + q_chunk, int(kappa.shape[0]))
                            kappa_chunk = kappa[q_start:q_end]
                            phase = dot[:, :, :, None] * kappa_chunk[None, None, None, :]
                            cos_p = torch.cos(phase)
                            sin_p = torch.sin(phase)
                            # Fuse phi_k * v into the contraction to avoid a large
                            # temporary kv tensor with shape (B, num_nodes, H, d, C).
                            moment_real = torch.einsum("blst,blhd,blhc->bsthdc", cos_p, phi_k, v_compute)
                            moment_imag = -torch.einsum("blst,blhd,blhc->bsthdc", sin_p, phi_k, v_compute)
                            b_real = torch.einsum("blhd,bsthdc->blsthc", phi_q, moment_real)
                            b_imag = torch.einsum("blhd,bsthdc->blsthc", phi_q, moment_imag)
                            if not self.a_per_l:
                                a_coeff = a_coeff_list[0]
                                if a_mode == "scalar":
                                    scale_chunk = a_coeff[q_start:q_end]
                                    acc_real_shc = (
                                        torch.einsum("blst,blsthc,t->blshc", cos_p, b_real, scale_chunk)
                                        - torch.einsum("blst,blsthc,t->blshc", sin_p, b_imag, scale_chunk)
                                    )
                                    acc_imag_shc = (
                                        torch.einsum("blst,blsthc,t->blshc", cos_p, b_imag, scale_chunk)
                                        + torch.einsum("blst,blsthc,t->blshc", sin_p, b_real, scale_chunk)
                                    )
                                elif a_mode == "head":
                                    scale_chunk = a_coeff[:, q_start:q_end]
                                    acc_real_shc = (
                                        torch.einsum("blst,blsthc,ht->blshc", cos_p, b_real, scale_chunk)
                                        - torch.einsum("blst,blsthc,ht->blshc", sin_p, b_imag, scale_chunk)
                                    )
                                    acc_imag_shc = (
                                        torch.einsum("blst,blsthc,ht->blshc", cos_p, b_imag, scale_chunk)
                                        + torch.einsum("blst,blsthc,ht->blshc", sin_p, b_real, scale_chunk)
                                    )
                                else:
                                    scale_chunk = a_coeff[:, :, q_start:q_end]
                                    if scale_chunk.shape[1] == 1 and C > 1:
                                        scale_chunk = scale_chunk.expand(-1, C, -1)
                                    acc_real_shc = (
                                        torch.einsum("blst,blsthc,hct->blshc", cos_p, b_real, scale_chunk)
                                        - torch.einsum("blst,blsthc,hct->blshc", sin_p, b_imag, scale_chunk)
                                    )
                                    acc_imag_shc = (
                                        torch.einsum("blst,blsthc,hct->blshc", cos_p, b_imag, scale_chunk)
                                        + torch.einsum("blst,blsthc,hct->blshc", sin_p, b_real, scale_chunk)
                                    )

                                for idx, W in enumerate(W_list):
                                    out_real_list[idx].add_(torch.einsum("blshc,sm->blhmc", acc_real_shc, W))
                                    out_imag_list[idx].add_(torch.einsum("blshc,sm->blhmc", acc_imag_shc, W))
                            else:
                                for idx, (W, a_coeff_l) in enumerate(zip(W_list, a_coeff_list)):
                                    if a_mode == "scalar":
                                        scale_chunk = a_coeff_l[q_start:q_end]
                                        acc_real_shc = (
                                            torch.einsum("blst,blsthc,t->blshc", cos_p, b_real, scale_chunk)
                                            - torch.einsum("blst,blsthc,t->blshc", sin_p, b_imag, scale_chunk)
                                        )
                                        acc_imag_shc = (
                                            torch.einsum("blst,blsthc,t->blshc", cos_p, b_imag, scale_chunk)
                                            + torch.einsum("blst,blsthc,t->blshc", sin_p, b_real, scale_chunk)
                                        )
                                    elif a_mode == "head":
                                        scale_chunk = a_coeff_l[:, q_start:q_end]
                                        acc_real_shc = (
                                            torch.einsum("blst,blsthc,ht->blshc", cos_p, b_real, scale_chunk)
                                            - torch.einsum("blst,blsthc,ht->blshc", sin_p, b_imag, scale_chunk)
                                        )
                                        acc_imag_shc = (
                                            torch.einsum("blst,blsthc,ht->blshc", cos_p, b_imag, scale_chunk)
                                            + torch.einsum("blst,blsthc,ht->blshc", sin_p, b_real, scale_chunk)
                                        )
                                    else:
                                        scale_chunk = a_coeff_l[:, :, q_start:q_end]
                                        if scale_chunk.shape[1] == 1 and C > 1:
                                            scale_chunk = scale_chunk.expand(-1, C, -1)
                                        acc_real_shc = (
                                            torch.einsum("blst,blsthc,hct->blshc", cos_p, b_real, scale_chunk)
                                            - torch.einsum("blst,blsthc,hct->blshc", sin_p, b_imag, scale_chunk)
                                        )
                                        acc_imag_shc = (
                                            torch.einsum("blst,blsthc,hct->blshc", cos_p, b_imag, scale_chunk)
                                            + torch.einsum("blst,blsthc,hct->blshc", sin_p, b_real, scale_chunk)
                                        )
                                    out_real_list[idx].add_(torch.einsum("blshc,sm->blhmc", acc_real_shc, W))
                                    out_imag_list[idx].add_(torch.einsum("blshc,sm->blhmc", acc_imag_shc, W))

                        out_list = []
                        for idx, l_mod in enumerate(self._l_mod_4):
                            out_real = out_real_list[idx]
                            out_imag = out_imag_list[idx]
                            if l_mod == 0:
                                out_acc = out_real
                            elif l_mod == 1:
                                out_acc = out_imag
                            elif l_mod == 2:
                                out_acc = -out_real
                            else:
                                out_acc = -out_imag
                            out_list.append((out_acc / Z[:, :, :, None, None]).to(dtype=v.dtype))

                    self.last_kappa_chunk = int(q_chunk)
                    break
                except RuntimeError as exc:
                    oom = "out of memory" in str(exc).lower()
                    if not (auto_chunk and oom and q_chunk > 1):
                        raise
                    if pos.device.type == "cuda":
                        torch.cuda.empty_cache()
                    q_chunk = max(1, q_chunk // 2)
        finally:
            if restore_tf32:
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                if prev_matmul_precision != self.matmul_precision:
                    torch.set_float32_matmul_precision(prev_matmul_precision)

        reshaped = [out.reshape(lead_shape + (H, 2 * l + 1, C)) for out, l in zip(out_list, self.l_values)]
        if not had_heads and q.ndim == pos.ndim and v.ndim == pos.ndim:
            reshaped = [out.squeeze(-3) for out in reshaped]
        return reshaped


class E2FormerAlphaFRYPrototype(nn.Module):
    """E2Former-aligned wrapper that builds Q/K/V projections and dispatches to a core operator.

    This is the main user-facing class. It bridges the E2Former feature representation
    (SO(3) irreps) with the alphaf(r)Y attention cores defined above.

    Data flow:
        1. **Q, K projections**: The full SO(3) irreps (all degrees 0..lmax) are
           "scalarized" via SO3_Linear2Scalar_e2former to produce per-head query and
           key vectors.  Shape: (B, num_nodes, (lmax+1)^2, C) -> (B, num_nodes, H*d) -> (B, num_nodes, H, d).
        2. **V projection**: Only the scalar (l=0) component of the irreps is used as
           input to a standard nn.Linear.  Shape: (B, num_nodes, C) -> (B, num_nodes, H*Cv) -> (B, num_nodes, H, Cv).
        3. **Core operator**: Q, K, V, and node positions are passed to either
           AlphaFRYSphericalSoftmax (exact, O(N^2)) or AlphaFRYSphericalFMM (approx, O(NSQ)).
        4. **Head merging**: The per-head outputs are optionally merged into a single
           channel dimension: (B, num_nodes, H, M, Cv) -> (B, num_nodes, M, H*Cv).
        5. **Embedding packing** (optional): The M=2l+1 components are placed back into
           a full (lmax+1)^2 irreps tensor at the correct degree-l slice.

    Expected inputs (matching this repo's E2Former layout):
      - node_pos:    (B, num_nodes, 3)  atomic/node positions
      - node_irreps: (B, num_nodes, (lmax+1)^2, C)  SO(3) feature tensor
        where index 0 along dim -2 corresponds to l=0 (scalar).

    Shape symbols:
      - B:  batch size
      - num_nodes:  number of nodes per graph
      - H:  num_heads
      - d:  head_dim (query/key dim per head)
      - Cv: value_dim (value dim per head)
      - l:  spherical harmonic degree for the output (set at init)
      - M:  2*l + 1 (number of m-components for degree l)

    Args:
        l:                Spherical harmonic degree for the output channel.
        kappa:            Radial frequencies, shape (Q,).
        a:                Radial mixing coefficients (see core classes for shape options).
        method:           "softmax" (exact O(N^2)) or "fmm" (linear + FMM, O(NSQ)).
        num_heads:        Number of attention heads H.
        head_dim:         Per-head query/key dimension d.
        value_dim:        Per-head value dimension Cv.
        num_directions:   Number of sphere quadrature points S (FMM only).
        sphere:           Quadrature scheme name (FMM only).
        phase_mode:       "complex" or "trig" (FMM only).
        kappa_chunk_size: Number of kappa frequencies processed per FMM chunk
                          (FMM only). Set 0 for auto chunk backoff mode.
        allow_tf32:       Enable TF32 fast matmul in FMM forward (CUDA float32).
        matmul_precision: torch float32 matmul precision ("highest"/"high"/"medium")
                          used when TF32 is enabled.
        promote_half_precision:
                          If True, promote fp16/bf16 inputs to fp32 in FMM compute
                          path (more stable). If False, keep native low precision
                          for higher throughput.
        optimize_low_precision_sphere:
                          If True, apply the low-precision sphere heuristic used by
                          `AlphaFRYSphericalFMM` (FMM only).
        feature_map:      Positive feature map for linear attention (FMM only).
        sh_backend:       Spherical-harmonics backend ("auto" | "e3nn" | "cueq").
        normalization:    e3nn spherical harmonic normalization convention.
        eps:              Numerical stability constant.
        lmax:             Maximum angular momentum in the input irreps (inferred if None).
        in_channels:      Number of feature channels C in the input irreps (inferred if None).
        merge_heads:      If True, merge H heads into the channel dim on output.
        return_embedding: If True, pack output into a full (lmax+1)^2 irreps tensor.
    """

    def __init__(
        self,
        *,
        l: int,
        kappa: torch.Tensor,
        a: torch.Tensor,
        method: str = "fmm",  # "softmax" | "fmm"
        num_heads: int = 8,
        head_dim: int = 32,
        value_dim: int = 32,
        num_directions: int = 64,
        sphere: str = "lebedev",
        phase_mode: str = "complex",
        kappa_chunk_size: int = 0,
        allow_tf32: bool = True,
        matmul_precision: str = "high",
        promote_half_precision: bool = True,
        optimize_low_precision_sphere: bool = True,
        feature_map: str = "elu",
        sh_backend: str = "auto",
        normalization: str = "integral",
        eps: float = 1e-8,
        lmax: Optional[int] = None,
        in_channels: Optional[int] = None,
        merge_heads: bool = True,
        return_embedding: bool = False,
    ) -> None:
        super().__init__()

        # --- Validate configuration ---
        if method not in ("softmax", "fmm"):
            raise ValueError(
                "method must be one of {'softmax','fmm'}, "
                f"got {method!r}. (Exact baselines moved to `molfm.models.e2former.fmm_exact`.)"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0 or value_dim <= 0:
            raise ValueError("head_dim and value_dim must be > 0")
        if (num_heads * head_dim) % 2 != 0:
            raise ValueError("num_heads*head_dim must be even (required by SO3_Linear2Scalar_e2former)")
        if return_embedding and not merge_heads:
            raise ValueError("return_embedding=True currently requires merge_heads=True")

        # Store configuration for later use in forward / projection building.
        self.l = int(l)
        self.method = method
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.value_dim = int(value_dim)
        self.merge_heads = bool(merge_heads)
        self.return_embedding = bool(return_embedding)

        # lmax and in_channels may be inferred lazily from the first forward call.
        self._lmax = int(lmax) if lmax is not None else None
        self._in_channels = int(in_channels) if in_channels is not None else None

        # --- Instantiate the core attention operator ---
        # The method flag selects the core:
        #   "softmax": AlphaFRYSphericalSoftmax  exact O(N^2) with standard softmax weights
        #   "fmm":     AlphaFRYSphericalFMM  linear attention + plane-wave factorization
        if method == "softmax":
            self.core = AlphaFRYSphericalSoftmax(
                l=l,
                kappa=kappa,
                a=a,
                normalization=normalization,
                sh_backend=sh_backend,
                eps=eps,
            )
        else:
            self.core = AlphaFRYSphericalFMM(
                l=l,
                kappa=kappa,
                a=a,
                num_directions=num_directions,
                sphere=sphere,
                phase_mode=phase_mode,
                kappa_chunk_size=kappa_chunk_size,
                allow_tf32=allow_tf32,
                matmul_precision=matmul_precision,
                promote_half_precision=promote_half_precision,
                optimize_low_precision_sphere=optimize_low_precision_sphere,
                sh_backend=sh_backend,
                feature_map=feature_map,
                normalization=normalization,
                eps=eps,
            )

        # --- Q/K/V projections (may be built lazily) ---
        # These are None until we know lmax and in_channels, either from init args
        # or from the first forward call.
        self.q_proj: Optional[nn.Module] = None
        self.k_proj: Optional[nn.Module] = None
        self.v_proj: Optional[nn.Module] = None
        if self._lmax is not None and self._in_channels is not None:
            self._build_projections(lmax=self._lmax, in_channels=self._in_channels)

    @staticmethod
    def _infer_lmax(l_sum: int) -> int:
        """Infer lmax from the packed irreps size.

        Packed irreps have size _{t=0}^{lmax}(2t+1) = (lmax+1)^2.
        So lmax = sqrt(l_sum) - 1.
        """
        root = int(math.isqrt(int(l_sum)))
        if root * root != int(l_sum):
            raise ValueError(f"Expected l_sum=(lmax+1)^2, got l_sum={l_sum}")
        return root - 1

    def _build_projections(
        self,
        *,
        lmax: int,
        in_channels: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Build the Q, K, V projection layers.

        - Q and K use SO3_Linear2Scalar_e2former, which takes the full SO(3) irreps
          (all degrees 0..lmax) and produces scalar features. This "scalarization"
          is specific to E2Former's design  it allows the attention mechanism to
          operate on scalar features while the input/output remain equivariant.

        - V uses a plain nn.Linear on the scalar (l=0) channel only.
          This is because values are multiplied by Y_l(r) inside the core operator,
          which reintroduces the angular dependence.
        """
        from .module_utils import SO3_Linear2Scalar_e2former  # local import to avoid circular deps

        out_dim = self.num_heads * self.head_dim
        self.q_proj = SO3_Linear2Scalar_e2former(in_channels, out_dim, lmax=lmax)
        self.k_proj = SO3_Linear2Scalar_e2former(in_channels, out_dim, lmax=lmax)
        # V projection: scalar channel -> H*Cv features.
        # Bias is included because this is a standard linear layer.
        self.v_proj = nn.Linear(in_channels, self.num_heads * self.value_dim, bias=True)
        if device is not None:
            self.q_proj = self.q_proj.to(device=device)
            self.k_proj = self.k_proj.to(device=device)
            self.v_proj = self.v_proj.to(device=device)
        self._lmax = int(lmax)
        self._in_channels = int(in_channels)

    def forward(
        self,
        node_pos: torch.Tensor,  # (B, num_nodes, 3)
        node_irreps: torch.Tensor,  # (B, num_nodes, (lmax+1)^2, C)
        *,
        node_mask: Optional[torch.Tensor] = None,  # (B, num_nodes) bool
    ) -> torch.Tensor:
        """Run the full E2Former alphaf(r)Y attention layer.

        Args:
            node_pos:    Atom/node positions, shape (B, num_nodes, 3).
            node_irreps: SO(3) irreps features, shape (B, num_nodes, (lmax+1)^2, C).
                         Dim -2 indexes the packed (l,m) components.
            node_mask:   Optional boolean mask, shape (B, num_nodes). True = valid node.

        Returns:
            If merge_heads=True and return_embedding=False:
                (B, num_nodes, M, H*Cv)  merged-head output for degree l.
            If merge_heads=False and return_embedding=False:
                (B, num_nodes, M, H, Cv)  per-head output for degree l.
            If return_embedding=True:
                (B, num_nodes, (lmax+1)^2, H*Cv)  full irreps-packed embedding.
        """
        # --- Input validation ---
        if node_pos.ndim != 3 or node_pos.shape[-1] != 3:
            raise ValueError(f"node_pos must be (B,num_nodes,3), got {tuple(node_pos.shape)}")
        if node_irreps.ndim != 4:
            raise ValueError(f"node_irreps must be (B,num_nodes,(lmax+1)^2,C), got {tuple(node_irreps.shape)}")
        if node_irreps.shape[:2] != node_pos.shape[:2]:
            raise ValueError("node_pos and node_irreps must share (B,num_nodes)")

        B, num_nodes = node_pos.shape[:2]
        l_sum = int(node_irreps.shape[-2])  # (lmax+1)^2
        in_channels = int(node_irreps.shape[-1])  # C
        lmax = self._infer_lmax(l_sum)

        # --- Lazy projection building ---
        # On the first forward call, we infer lmax and in_channels from the input
        # tensor shape and build the projection layers. This allows the user to
        # create the module without knowing these dimensions upfront.
        if self.q_proj is None or self.k_proj is None or self.v_proj is None:
            self._build_projections(lmax=lmax, in_channels=in_channels, device=node_irreps.device)
        else:
            # Consistency check: make sure input dims match what projections expect.
            if self._lmax != lmax or self._in_channels != in_channels:
                raise ValueError(
                    f"Projection modules were built for lmax={self._lmax},C={self._in_channels} "
                    f"but got lmax={lmax},C={in_channels}"
                )

        # --- Build Q, K, V ---
        # E2Former convention: run projections in fp32 even under AMP for stability.
        node_irreps_f = node_irreps.to(dtype=torch.float32)
        node_pos_f = node_pos  # the core operator handles compute dtype internally

        # Q and K from full irreps: (B,num_nodes,(lmax+1)^2,C) -> (B,num_nodes,H*d) -> (B,num_nodes,H,d)
        q = self.q_proj(node_irreps_f).view(B, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_irreps_f).view(B, num_nodes, self.num_heads, self.head_dim)

        # V from scalar (l=0) channel only: (B,num_nodes,C) -> (B,num_nodes,H*Cv) -> (B,num_nodes,H,Cv)
        # The l=0 component sits at index 0 in the packed (l,m) dimension.
        scalar = node_irreps_f[:, :, 0, :]  # (B, num_nodes, C)
        v = self.v_proj(scalar).view(B, num_nodes, self.num_heads, self.value_dim)

        # --- Core attention computation ---
        # Dispatch to the selected core (softmax or FMM).
        # The core receives Q, K, V as (B, num_nodes, H, d/Cv) and positions as (B, num_nodes, 3).
        # It returns either (B, num_nodes, H, M, Cv) or (B, num_nodes, M, Cv).
        out = self.core(node_pos_f, q, k, v, node_mask=node_mask)

        # --- Post-processing: head merging ---
        if out.ndim == 4:
            # Core returned unheaded output: (B, num_nodes, M, Cv)
            out_m = out
        else:
            # Core returned headed output: (B, num_nodes, H, M, Cv)
            # Permute to (B, num_nodes, M, H, Cv) so that the H and Cv dims are adjacent,
            # making the subsequent head-merge into (B, num_nodes, M, H*Cv) a contiguous view.
            out = out.permute(0, 1, 3, 2, 4).contiguous()  # (B, num_nodes, M, H, Cv)
            if self.merge_heads:
                # Concatenate all heads: (B, num_nodes, M, H*Cv)
                out_m = out.view(B, num_nodes, 2 * self.l + 1, self.num_heads * self.value_dim)
            else:
                out_m = out  # (B, num_nodes, M, H, Cv)  keep heads separate

        if not self.return_embedding:
            return out_m

        # --- Embedding packing (optional) ---
        # Place the degree-l output into a full (lmax+1)^2 irreps tensor.
        # All other degrees are zero-filled. This allows stacking outputs from
        # multiple l values into a single irreps representation.
        if self._lmax is None:
            raise RuntimeError("Internal error: lmax unknown for embedding packing")
        out_channels = out_m.shape[-1]
        emb = torch.zeros(
            (B, num_nodes, (self._lmax + 1) ** 2, out_channels),
            device=out_m.device,
            dtype=out_m.dtype,
        )
        # Degree l occupies index range [l^2, (l+1)^2) in the packed (l,m) axis.
        # For example: l=0 -> [0,1), l=1 -> [1,4), l=2 -> [4,9), etc.
        start = self.l * self.l
        end = (self.l + 1) * (self.l + 1)
        emb[:, :, start:end, :] = out_m
        return emb
