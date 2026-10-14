"""
Helpers for atomic positions under periodic boundary conditions.

Positions are rows, pos = frac @ L, where the rows of the cell L are the
lattice vectors. Fractional coordinates live on the unit torus [0, 1)^3.

The conversions run with autocast disabled and in at least fp32: under bf16
the fractional coordinates would be off by ~1e-2, i.e. ~0.1 Å for a 10 Å cell.
"""

import torch

from mlip_distillation.common.lattice_decomposition import (
    _linalg_upcast_dtype,
    safe_inverse,
)


def wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap a fractional displacement into [-1/2, 1/2], i.e. the shortest
    signed displacement on the unit torus."""
    return d - torch.round(d)


def _compute_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    return torch.promote_types(
        _linalg_upcast_dtype(a.dtype), _linalg_upcast_dtype(b.dtype)
    )


def cart_to_frac(
    pos: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """pos: [n_atoms, 3], cell: [n_structures, 3, 3], batch: [n_atoms] -> frac = pos @ L^-1"""
    dtype = _compute_dtype(pos, cell)
    with torch.autocast(device_type=pos.device.type, enabled=False):
        cell_inv = safe_inverse(cell.to(dtype))
        return torch.einsum("ni,nij->nj", pos.to(dtype), cell_inv[batch])


def frac_to_cart(
    frac: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """frac: [n_atoms, 3], cell: [n_structures, 3, 3], batch: [n_atoms] -> pos = frac @ L"""
    dtype = _compute_dtype(frac, cell)
    with torch.autocast(device_type=frac.device.type, enabled=False):
        return torch.einsum("ni,nij->nj", frac.to(dtype), cell.to(dtype)[batch])
