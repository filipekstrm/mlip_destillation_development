import math
import os

import numpy as np
import torch
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from torch.amp import autocast

from mlip_distillation.distillation.flow_map_trainer import FlowMapTrainer
from mlip_distillation.distributed_utils import is_main_process


class LSDTrainer(FlowMapTrainer):
    def train(self, flow_map, eta, optimizer, num_epochs, device):
        losses = []
        losses_b = []
        losses_lsd = []
        print("Starting training flow map with LSD")
        for epoch in range(num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for x0, x1 in self.train_loader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                optimizer.zero_grad()

                M_d = math.floor(eta * len(x1))
                perm = torch.randperm(x1.cell.size(0))

                i_idx = perm[:M_d]
                x1_i = atomicdata_list_to_batch(x1[i_idx])
                x0_i = atomicdata_list_to_batch(x0[i_idx])
                t_i = torch.rand((M_d,), device=device)
                It_i, cell_t_i_aux = self.build_stochastic_interpolant(x0_i, x1_i, t_i)
                It_i_dot = self.stochastic_interpolant_derivative(
                    x0_i, x1_i, cell_t_i_aux
                )
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

                    bt_j_pos, bt_j_cell = flow_map.v(Xst, t_j, t_j)
                    r_j_pos = dXst_dt_pos - bt_j_pos.detach()
                    r_j_cell = dXst_dt_cell - bt_j_cell.detach()
                    w_st = flow_map.compute_weight(s_j, t_j)
                    L_LSD = self.compute_loss(
                        w_st, torch.sum(r_j_pos**2, dim=-1, keepdim=True)
                    ) + self.compute_loss(w_st, torch.sum(r_j_cell**2, dim=(-1, -2)))

                    L_SD = L_b + L_LSD
                L_SD.backward()
                # torch.nn.utils.clip_grad_norm_(flow_map.parameters(), 10)
                optimizer.step()
                losses_lsd.append(L_LSD.item())
                losses_b.append(L_b.item())
                losses.append(L_SD.item())
            print(
                f"Epoch {epoch + 1} loss:\n",
                f"L_B loss: {np.mean(losses_b)}\n",
                f"L_LSD loss: {np.mean(losses_lsd)}",
            )

            if is_main_process():
                self.save_checkpoint(flow_map, epoch)
        return flow_map

    def save_checkpoint(self, model, epoch):
        # unwrap DDP if present, otherwise save as-is
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        torch.save(
            state_dict, os.path.join("checkpoints", f"checkpoint_epoch={epoch}.pt")
        )
