"""Pure-PyTorch fallback for the `torch_scatter` API subset used here."""

from __future__ import annotations

from typing import Optional

import torch

from molfm.utils.pyg_fallback import (
    scatter,
    scatter_mean,
    scatter_softmax,
    scatter_sum,
)

__all__ = [
    "scatter",
    "scatter_add",
    "scatter_sum",
    "scatter_mean",
    "scatter_softmax",
    "segment_coo",
    "segment_csr",
]

__version__ = "0.0.0-fallback"


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter(src=src, index=index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Segment reduction where `index` gives the segment id per row."""
    return scatter(
        src=src,
        index=index,
        dim=0,
        out=out,
        dim_size=dim_size,
        reduce=reduce,
    )


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Segment reduction over CSR pointers on the first dimension."""
    if src.dim() == 1:
        src_2d = src.unsqueeze(-1)
        squeeze = True
    else:
        src_2d = src
        squeeze = False

    n_seg = int(indptr.numel()) - 1
    if out is None:
        out_2d = torch.zeros(
            (n_seg,) + tuple(src_2d.shape[1:]),
            device=src_2d.device,
            dtype=src_2d.dtype,
        )
    else:
        out_2d = out.clone()
        if out_2d.dim() == 1:
            out_2d = out_2d.unsqueeze(-1)

    for s in range(n_seg):
        start = int(indptr[s].item())
        end = int(indptr[s + 1].item())
        if end <= start:
            continue
        chunk = src_2d[start:end]
        mode = "sum" if reduce == "add" else reduce
        if mode == "sum":
            out_2d[s] = chunk.sum(dim=0)
        elif mode == "mean":
            out_2d[s] = chunk.mean(dim=0)
        elif mode in {"max", "amax"}:
            out_2d[s] = chunk.max(dim=0).values
        elif mode in {"min", "amin"}:
            out_2d[s] = chunk.min(dim=0).values
        else:
            raise ValueError(f"Unsupported reduce mode for segment_csr: {reduce}")

    if squeeze:
        out_2d = out_2d.squeeze(-1)
    return out_2d
