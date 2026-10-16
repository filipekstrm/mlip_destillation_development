import torch
import torch.nn as nn

from mlip_distillation.common.periodic import cart_to_frac, frac_to_cart
from mlip_distillation.models import utils
from mlip_distillation.models.velocity_field import VelocityField


class FlowMap(nn.Module):
    def __init__(self, model, learnable_weights_dim=None):
        super().__init__()
        self.v = VelocityField(model)
        if learnable_weights_dim is not None:
            self.weights_model = utils.WeightsMLP(learnable_weights_dim)
        else:
            self.weights_model = None

    def compute_weight(self, s, t):
        if self.weights_model is not None:
            return self.weights_model(s, t)
        else:
            return torch.as_tensor(0.0)

    def step(self, data, s, t):
        """
        Positions are stepped in fractional coordinates, so that the predicted
        dpos is the velocity relative to the cell (in Cartesian units):
        f_st = f_s + (t - s) * dpos @ L_s^-1 (wrapped into [0, 1)),
        L_st = L_s + (t - s) * dlattice. Returns (f_st, L_st).
        """
        device = data.pos.device
        B = data.cell.shape[0]
        t = utils.expand_time(t, B).to(device)
        s = utils.expand_time(s, B).to(device)
        dpos, dlattice = self.v(data, s, t)
        dt = t - s
        frac = cart_to_frac(data.pos, data.cell, data.batch)
        dfrac = cart_to_frac(dpos, data.cell, data.batch)
        new_frac = torch.remainder(
            frac + dt.repeat_interleave(data.natoms).view(-1, 1) * dfrac, 1.0
        )
        new_cell = data.cell + dt.view(-1, 1, 1) * dlattice
        return new_frac, new_cell

    def forward(self, data, s, t):
        new_frac, new_cell = self.step(data, s, t)
        data_new = data.clone()
        data_new.pos = frac_to_cart(new_frac, new_cell, data.batch).to(data.pos.dtype)
        data_new.cell = new_cell
        return data_new

    def partial_t(self, data, s, t):
        """
        Returns (X_st, d/dt of the positions relative to the cell, dL_st/dt).
        The position derivative is df_st/dt @ L_st, which is in the same units
        as the position output of v.
        """
        device = data.pos.device
        B = data.cell.shape[0]
        t = utils.expand_time(t, B).to(device)
        s = utils.expand_time(s, B).to(device)
        (
            (frac_st, cell_st),
            (dfrac_st_dt, dcell_st_dt),
        ) = torch.autograd.functional.jvp(  # use torch.func.jvp if possible
            lambda t: self.step(data, s, t),
            inputs=(t,),
            v=(torch.ones_like(t),),
            create_graph=True,
        )
        Xst = data.clone()
        Xst.pos = frac_to_cart(frac_st, cell_st, data.batch).to(data.pos.dtype)
        Xst.cell = cell_st
        dpos_st_dt = frac_to_cart(dfrac_st_dt, cell_st, data.batch)
        return Xst, dpos_st_dt, dcell_st_dt
