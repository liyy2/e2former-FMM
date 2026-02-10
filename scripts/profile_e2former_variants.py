# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kernel-level torch.profiler for E2former variants on synthetic inputs "
            "(baseline edge attention vs fmm-node vs hybrid)."
        )
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--B", type=int, default=2, help="Batch size (graphs).")
    parser.add_argument(
        "--nodes-per-graph",
        "--num-nodes",
        dest="nodes_per_graph",
        type=int,
        default=512,
        help="Number of nodes per graph.",
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of Transformer blocks.")
    parser.add_argument("--max-neighbors", type=int, default=20, help="Neighbor cap for baseline.")
    parser.add_argument("--radius", type=float, default=15.0, help="Cutoff radius for baseline.")
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=0.1,
        help="Scale of random positions. Smaller values increase baseline neighborhood density.",
    )
    parser.add_argument("--number-of-basis", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attn-scalar-head", type=int, default=32)
    parser.add_argument(
        "--fmm-tp-backend",
        type=str,
        default="cueq",
        choices=["auto", "e3nn", "cueq"],
        help="Tensor-product backend tag for fmm-node attention.",
    )
    parser.add_argument("--fmm-num-kappa", type=int, default=6)
    parser.add_argument("--fmm-kappa-min", type=float, default=0.8)
    parser.add_argument("--fmm-kappa-max", type=float, default=1.2)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,fmm-node,hybrid",
        help="Comma-separated list of variants to profile.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Include backward pass (profiles more than just attention).",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before profiling.")
    parser.add_argument("--active", type=int, default=4, help="Number of profiled iterations.")
    parser.add_argument(
        "--row-limit",
        type=int,
        default=30,
        help="Rows to print from profiler table.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="",
        help="Optional directory to write chrome traces (one subdir per variant).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _loss_from_outputs(outputs: object) -> torch.Tensor:
    if isinstance(outputs, tuple):
        loss = torch.tensor(0.0, device=outputs[0].device)
        for item in outputs:
            if torch.is_tensor(item):
                loss = loss + item.float().square().mean()
        return loss
    if torch.is_tensor(outputs):
        return outputs.float().square().mean()
    raise TypeError(f"Unexpected model output type: {type(outputs)}")


def _make_models(*, args: argparse.Namespace, device: torch.device) -> dict[str, torch.nn.Module]:
    from molfm.models.e2former.e2former import E2former  # noqa: WPS433

    common = dict(
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=args.layers,
        pbc_max_radius=float(args.radius),
        max_radius=float(args.radius),
        basis_type="gaussiansmear",
        number_of_basis=args.number_of_basis,
        num_attn_heads=args.heads,
        attn_scalar_head=args.attn_scalar_head,
        irreps_head="32x0e+32x1e+32x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="rms_norm_sh_BL",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        atom_type_cnt=256,
        edge_embedtype="eqv2",
        attn_biastype="share",
        ffn_type="eqv2ffn",
        add_rope=False,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        force_head=None,
        decouple_EF=False,
        max_neighbors=args.max_neighbors,
    )

    if args.fmm_tp_backend == "e3nn":
        fmm_tp_type = "fmm-node+tp_e3nn"
    elif args.fmm_tp_backend == "cueq":
        fmm_tp_type = "fmm-node+tp_cueq"
    else:
        fmm_tp_type = "fmm-node"

    models: dict[str, torch.nn.Module] = {}
    models["baseline"] = E2former(tp_type="QK_alpha", attn_type="first-order", **common).to(device)
    models["fmm-node"] = E2former(
        tp_type=fmm_tp_type,
        attn_type="fmm-node",
        fmm_num_kappa=args.fmm_num_kappa,
        fmm_kappa_min=args.fmm_kappa_min,
        fmm_kappa_max=args.fmm_kappa_max,
        **common,
    ).to(device)
    models["hybrid"] = E2former(
        tp_type=f"QK_alpha@{fmm_tp_type}",
        attn_type="hybrid-first-order",
        fmm_num_kappa=args.fmm_num_kappa,
        fmm_kappa_min=args.fmm_kappa_min,
        fmm_kappa_max=args.fmm_kappa_max,
        **common,
    ).to(device)
    return models


def _make_inputs(*, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    B = int(args.B)
    num_nodes = int(args.nodes_per_graph)
    pos = float(args.pos_scale) * torch.randn(B, num_nodes, 3, device=device)
    masked_token_type = torch.randint(1, 20, (B, num_nodes), device=device)
    padding_mask = torch.zeros(B, num_nodes, dtype=torch.bool, device=device)
    pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)
    batched_data = {"pos": pos, "masked_token_type": masked_token_type, "pbc": pbc}
    return batched_data, padding_mask


def _profile_variant(
    *,
    name: str,
    model: torch.nn.Module,
    batched_data: dict[str, torch.Tensor],
    padding_mask: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    if args.backward:
        model.train()
    else:
        model.eval()

    for _ in range(int(args.warmup)):
        outputs = model(batched_data=batched_data, token_embedding=None, padding_mask=padding_mask)
        if args.backward:
            _loss_from_outputs(outputs).backward()
            model.zero_grad(set_to_none=True)
    _maybe_sync(device)

    trace_dir = None
    if args.trace_dir:
        trace_dir = Path(args.trace_dir).expanduser().resolve() / name
        trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _trace_handler(prof: torch.profiler.profile) -> None:
        if trace_dir is None:
            return
        prof.export_chrome_trace(str(trace_dir / "trace.json"))

    schedule_cfg = torch.profiler.schedule(wait=0, warmup=0, active=int(args.active), repeat=1)
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule_cfg,
        on_trace_ready=_trace_handler,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:
        for _ in range(int(args.active)):
            outputs = model(batched_data=batched_data, token_embedding=None, padding_mask=padding_mask)
            if args.backward:
                _loss_from_outputs(outputs).backward()
                model.zero_grad(set_to_none=True)
            prof.step()
    _maybe_sync(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(f"\n== torch.profiler: {name} (backward={args.backward}) ==")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=int(args.row_limit)))


def main() -> None:
    args = _parse_args()
    _add_src_to_path()

    torch.manual_seed(int(args.seed))
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    requested = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    models = _make_models(args=args, device=device)
    batched_data, padding_mask = _make_inputs(args=args, device=device)

    for name in requested:
        if name not in models:
            raise ValueError(f"Unknown variant {name!r}. Available: {sorted(models)}")
        _profile_variant(
            name=name,
            model=models[name],
            batched_data=batched_data,
            padding_mask=padding_mask,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()

