import torch
import torch.nn as nn

from mlip_distillation.models import utils


class FlowMap(nn.Module):
    def __init__(self, model, learnable_weights_dim=None):
        super().__init__()
        self.v_model = model
        if learnable_weights_dim is not None:
            self.weights_model = utils.WeightsMLP(learnable_weights_dim)
        else:
            self.weights_model = None

    def v(self, data, s, t):
        dpos, dlattice_unscaled = self.v_model(data, s, t)
        V = torch.linalg.det(data.cell).view(-1, 1, 1)
        dlattice = V * torch.matmul(
            dlattice_unscaled, torch.inverse(data.cell).transpose(-1, -2)
        )
        return dpos, dlattice

    def compute_weight(self, s, t):
        if self.weights_model is not None:
            return self.weights_model(s, t)
        else:
            return torch.as_tensor(0.0)

    def forward(self, data, s, t, only_tensors=False):
        device = data.pos.device
        B = data.cell.shape[0]
        t = utils.expand_time(t, B).to(device)
        s = utils.expand_time(s, B).to(device)
        dpos, dlattice = self.v(data, s, t)
        dt = t - s
        new_pos = data.pos + dt.repeat_interleave(data.natoms).view(-1, 1) * dpos
        new_cell = data.cell + dt.view(-1, 1, 1) * dlattice
        if only_tensors:
            return new_pos, new_cell
        data_new = data.clone()
        data_new.pos = new_pos
        data_new.cell = new_cell
        return data_new

    def partial_t(self, data, s, t):
        device = data.pos.device
        B = data.cell.shape[0]
        t = utils.expand_time(t, B).to(device)
        s = utils.expand_time(s, B).to(device)
        (
            Xst_tuple,
            dt_Xst,
        ) = torch.autograd.functional.jvp(  # use torch.func.jvp if possible
            lambda t: self(data, s, t, only_tensors=True),
            inputs=(t,),
            v=(torch.ones_like(t),),
            create_graph=True,
        )
        Xst = data.clone()
        Xst.pos = Xst_tuple[0]
        Xst.cell = Xst_tuple[1]
        return Xst, *dt_Xst
