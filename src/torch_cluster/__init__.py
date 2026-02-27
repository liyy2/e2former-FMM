"""Pure-PyTorch fallback for the `torch_cluster` API subset used here."""

from molfm.utils.pyg_fallback import radius_graph

__all__ = ["radius_graph"]
__version__ = "0.0.0-fallback"
