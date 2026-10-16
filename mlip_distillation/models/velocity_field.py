import torch
import torch.nn as nn


class VelocityField(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data, s, t=None):
        dpos, strain_rate = self.model(data, s, t)
        # Cell moves by a pure (symmetric) strain, no rotation: dL = L @ sigma
        # (row convention, rows of cell are lattice vectors)
        strain_rate = 0.5 * (strain_rate + strain_rate.transpose(-1, -2))
        dlattice = torch.matmul(data.cell, strain_rate)
        return dpos, dlattice
