from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlip_distillation.common.lattice_decomposition import (
    compute_Lt,
    compute_Lt_dot,
    fit_transform,
)
from mlip_distillation.common.periodic import cart_to_frac, frac_to_cart, wrap


class FlowMapTrainer(ABC):
    def __init__(self, dataloader, sampler=None):
        self.train_loader = dataloader
        self.sampler = sampler

    @torch.no_grad()
    def derotate(self, x0, x1):
        """
        Rigidly rotate x1 (cell and positions) by R^T, where L0^-1 L1 = R @ S
        is the polar decomposition. Afterwards L0^-1 L1 is symmetric, i.e. the
        cells differ by a pure stretch. The energy is rotation invariant, so this
        only fixes the orientation of the relaxed structure.
        """
        R = fit_transform(x0.cell, x1.cell)["R"]
        x1 = x1.clone()
        x1.cell = x1.cell @ R.transpose(-1, -2)
        x1.pos = torch.einsum("nj,nkj->nk", x1.pos, R[x1.batch])
        return x1

    @torch.no_grad()
    def build_stochastic_interpolant(self, x0, x1, t):
        """
        Positions follow the geodesic on the torus of fractional coordinates,
        f_t = f0 + t * wrap(f1 - f0) (wrapped into [0, 1)), and are mapped back
        with the interpolated cell, pos_t = f_t @ L_t.
        """
        It = x0.clone()
        t_pos = t.repeat_interleave(x0.natoms, dim=0).view(-1, 1)
        t_cell = t.view(-1)
        f0 = cart_to_frac(x0.pos, x0.cell, x0.batch)
        f1 = cart_to_frac(x1.pos, x1.cell, x1.batch)
        dfrac = wrap(f1 - f0)
        ft = torch.remainder(f0 + t_pos * dfrac, 1.0)
        It.cell, cell_aux = compute_Lt(x0.cell, x1.cell, t_cell, remove_rotation=True)
        It.pos = frac_to_cart(ft, It.cell, x0.batch).to(x0.pos.dtype)
        return It, {"cell": cell_aux, "dfrac": dfrac}

    @torch.no_grad()
    def stochastic_interpolant_derivative(self, x0, It, aux):
        """
        The position target is the velocity relative to the cell in Cartesian
        units, df_t/dt @ L_t = wrap(f1 - f0) @ L_t, matching the position output
        of FlowMap.v. The motion from the deforming cell is not included: FlowMap
        adds it by stepping in fractional coordinates.
        """
        It_dot = x0.clone()
        It_dot.pos = frac_to_cart(aux["dfrac"], It.cell, It.batch).to(x0.pos.dtype)
        It_dot.cell = compute_Lt_dot(x0.cell, aux["cell"])
        return It_dot

    def sample_s_t(self, batch_size, device):
        s = torch.rand((batch_size, 1), device=device)
        t = s + torch.rand_like(s) * (1 - s)
        return s, t

    @abstractmethod
    def train(self, flow_map, optimizer, num_epochs):
        pass

    def compute_loss(
        self,
        weight,
        sq_norm,
    ):
        return torch.mean(torch.exp(-weight) * sq_norm + weight)
