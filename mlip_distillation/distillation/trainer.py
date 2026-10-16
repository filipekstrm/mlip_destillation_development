from abc import ABC, abstractmethod
from collections import defaultdict

import torch

from mlip_distillation.common.lattice_decomposition import (
    compute_Lt,
    compute_Lt_dot,
    fit_transform,
)
from mlip_distillation.common.periodic import cart_to_frac, frac_to_cart, wrap
from mlip_distillation.distributed_utils import global_mean, is_main_process


class Trainer(ABC):
    def __init__(self, dataloader, logger, sampler=None):
        self.train_loader = dataloader
        self.logger = logger
        self.sampler = sampler

    def save_checkpoint(self, model, optimizer, epoch):
        # Only the main process saves. Unwrap DDP if present.
        if not is_main_process():
            return
        model = model.module if hasattr(model, "module") else model
        self.logger.save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            best=False,
            epoch=epoch,
        )

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

    @abstractmethod
    def training_step(self, model, x0, x1, device):
        """
        Loss on one batch of (initial, derotated relaxed) pairs. Returns the
        loss to backpropagate and a dict of scalar tensors to log. Must return
        the same keys on every call and every rank.
        """
        pass

    def train(self, model, optimizer, num_epochs, device, start_epoch=0):
        if is_main_process():
            print(f"Starting training with {type(self).__name__}")
        for epoch in range(start_epoch, num_epochs):
            step_logs = defaultdict(list)
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for x0, x1 in self.train_loader:
                x0 = x0.to(device)
                x1 = self.derotate(x0, x1.to(device))
                optimizer.zero_grad()
                loss, logs = self.training_step(model, x0, x1, device)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                for key, value in logs.items():
                    step_logs[key].append(value.item())
            # Epoch means over all ranks (all ranks must take part in the reduce)
            epoch_log = {key: global_mean(v, device) for key, v in step_logs.items()}
            if is_main_process():
                print(
                    f"Epoch {epoch + 1} loss:\n"
                    + "\n".join(f"  {key}: {v}" for key, v in epoch_log.items())
                )
                self.logger.log(epoch_log, step=epoch)
            self.save_checkpoint(model, optimizer, epoch)
        return model
