import math

import torch
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from torch.amp import autocast

from mlip_distillation.distillation.flow_map_trainer import FlowMapTrainer


class LSDTrainer(FlowMapTrainer):
    def __init__(self, dataloader, logger, eta, sampler=None):
        super().__init__(dataloader, logger, sampler)
        self.eta = eta

    def training_step(self, flow_map, x0, x1, device):
        M_d = math.floor(self.eta * len(x1))
        perm = torch.randperm(x1.cell.size(0))

        i_idx = perm[:M_d]
        x1_i = atomicdata_list_to_batch(x1[i_idx])
        x0_i = atomicdata_list_to_batch(x0[i_idx])
        t_i = torch.rand((M_d,), device=device)
        It_i, aux_i = self.build_stochastic_interpolant(x0_i, x1_i, t_i)
        It_i_dot = self.stochastic_interpolant_derivative(x0_i, It_i, aux_i)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            w_ti = flow_map.compute_weight(t_i, t_i)
            pos_i_pred, cell_i_pred = flow_map.v(It_i, t_i, t_i)
            pos_b_diff = torch.sum(
                (pos_i_pred - It_i_dot.pos) ** 2, dim=-1, keepdim=True
            )
            cell_b_diff = torch.sum(
                (cell_i_pred - It_i_dot.cell) ** 2, dim=(-1, -2)
            ).view(-1, 1)
            L_b = self.compute_loss(w_ti, pos_b_diff) + self.compute_loss(
                w_ti, cell_b_diff
            )

            M_o = len(x1) - M_d
            j_idx = perm[M_d:]
            x1_j = atomicdata_list_to_batch(x1[j_idx])
            x0_j = atomicdata_list_to_batch(x0[j_idx])
            s_j, t_j = self.sample_s_t(M_o, device)
            Is_j, _ = self.build_stochastic_interpolant(x0_j, x1_j, s_j)
            Xst, dXst_dt_pos, dXst_dt_cell = flow_map.partial_t(Is_j, s_j, t_j)

            # Stop-gradient on the target. Equivalent to calling .detach() on
            # the results, but no graph is recorded, which saves memory.
            # Eval mode makes the target the full network, without dropout,
            # instead of a pass with different random masks than partial_t.
            with torch.no_grad():
                flow_map.eval()
                bt_j_pos, bt_j_cell = flow_map.v(Xst, t_j, t_j)
                flow_map.train()
            r_j_pos = dXst_dt_pos - bt_j_pos
            r_j_cell = dXst_dt_cell - bt_j_cell
            w_st = flow_map.compute_weight(s_j, t_j)
            L_LSD = self.compute_loss(
                w_st, torch.sum(r_j_pos**2, dim=-1, keepdim=True)
            ) + self.compute_loss(w_st, torch.sum(r_j_cell**2, dim=(-1, -2)))

            L_SD = L_b + L_LSD
        return L_SD, {"loss": L_SD, "L_B": L_b, "L_LSD": L_LSD}
