"""Fallback implementations for minimal PyG ops used in this repository.

These functions are used only when native `torch_scatter` / `torch_cluster`
extensions are unavailable at runtime.
"""

from __future__ import annotations

from typing import Optional

import torch


def _normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    dim = _normalize_dim(dim, src.dim())
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(*shape)
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)
    return index.expand_as(src)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Lightweight `torch_scatter.scatter` substitute."""
    dim = _normalize_dim(dim, src.dim())
    index_exp = _expand_index(index, src, dim)

    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    else:
        out = out.clone()

    reduce = "sum" if reduce == "add" else reduce
    if reduce == "sum":
        return out.scatter_add(dim, index_exp, src)

    if reduce == "mean":
        out = out.scatter_add(dim, index_exp, src)
        count = torch.zeros_like(out)
        ones = torch.ones_like(src)
        count = count.scatter_add(dim, index_exp, ones)
        return out / count.clamp_min(1)

    if reduce in {"max", "amax"}:
        init = out.fill_(float("-inf"))
        return init.scatter_reduce(dim, index_exp, src, reduce="amax", include_self=True)

    if reduce in {"min", "amin"}:
        init = out.fill_(float("inf"))
        return init.scatter_reduce(dim, index_exp, src, reduce="amin", include_self=True)

    raise ValueError(f"Unsupported reduce mode: {reduce}")


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(
    src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> torch.Tensor:
    """Stable softmax over segments defined by `index`."""
    dim = _normalize_dim(dim, src.dim())
    index_exp = _expand_index(index, src, dim)

    max_per_group = scatter(
        src=src, index=index, dim=dim, dim_size=dim_size, reduce="max"
    )
    max_gather = max_per_group.gather(dim, index_exp)
    numer = torch.exp(src - max_gather)

    denom_per_group = scatter(
        src=numer, index=index, dim=dim, dim_size=dim_size, reduce="sum"
    )
    denom_gather = denom_per_group.gather(dim, index_exp)
    return numer / denom_gather.clamp_min(1e-12)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: Optional[int] = 32,
) -> torch.Tensor:
    """Fallback `radius_graph` implementation using `torch.cdist`.

    Returns edge_index in source-to-target format with shape [2, E].
    """
    if x.dim() != 2:
        raise ValueError(f"`x` must have shape [N, F], got {tuple(x.shape)}.")

    n_nodes = x.size(0)
    if batch is None:
        batch = torch.zeros(n_nodes, dtype=torch.long, device=x.device)
    else:
        batch = batch.long().to(x.device)

    src_chunks = []
    dst_chunks = []

    for b in batch.unique(sorted=True):
        node_ids = torch.nonzero(batch == b, as_tuple=False).flatten()
        if node_ids.numel() == 0:
            continue
        pos = x[node_ids]
        dmat = torch.cdist(pos, pos)
        if not loop:
            dmat.fill_diagonal_(float("inf"))

        for i_local in range(node_ids.numel()):
            nbr_local = torch.nonzero(dmat[:, i_local] <= r, as_tuple=False).flatten()
            if nbr_local.numel() == 0:
                continue
            if max_num_neighbors is not None and max_num_neighbors > 0:
                if nbr_local.numel() > max_num_neighbors:
                    nbr_d = dmat[nbr_local, i_local]
                    keep = torch.topk(
                        nbr_d, k=max_num_neighbors, largest=False
                    ).indices
                    nbr_local = nbr_local[keep]

            src_chunks.append(node_ids[nbr_local])
            dst_chunks.append(
                torch.full(
                    (nbr_local.numel(),),
                    int(node_ids[i_local].item()),
                    device=x.device,
                    dtype=torch.long,
                )
            )

    if not src_chunks:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    src = torch.cat(src_chunks, dim=0)
    dst = torch.cat(dst_chunks, dim=0)
    return torch.stack((src, dst), dim=0)
