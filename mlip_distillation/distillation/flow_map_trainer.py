from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlip_distillation.common.lattice_decomposition import compute_Lt, compute_Lt_dot


class FlowMapTrainer(ABC):
    def __init__(self, dataloader, sampler=None):
        self.train_loader = dataloader
        self.sampler = sampler

    @torch.no_grad()
    def build_stochastic_interpolant(self, x0, x1, t):
        It = x0.clone()
        t_pos = t.repeat_interleave(x0.natoms, dim=0).view(-1, 1)
        t_cell = t.view(-1)
        It.pos = (1 - t_pos) * x0.pos + t_pos * x1.pos
        It.cell, cell_aux = compute_Lt(x0.cell, x1.cell, t_cell)
        return It, cell_aux

    @torch.no_grad()
    def stochastic_interpolant_derivative(self, x0, x1, cell_aux):
        It_dot = x0.clone()
        It_dot.pos = x1.pos - x0.pos
        It_dot.cell = compute_Lt_dot(x0.cell, cell_aux)
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
