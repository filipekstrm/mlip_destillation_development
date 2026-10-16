import torch
from torch.amp import autocast

from mlip_distillation.distillation.trainer import Trainer


class FlowMatchingTrainer(Trainer):
    """
    Naive flow matching: regress v(I_t, t, t) onto dI_t/dt with an unweighted
    squared error. The model is a FlowMap evaluated on the diagonal s = t, so
    its checkpoints can also initialize LSD training.
    """

    def sample_t(self, batch_size, device):
        return torch.rand((batch_size,), device=device)

    def compute_loss(self, sq_norm):
        return torch.mean(sq_norm)

    def training_step(self, model, x0, x1, device):
        t = self.sample_t(len(x1), device)
        It, aux = self.build_stochastic_interpolant(x0, x1, t)
        It_dot = self.stochastic_interpolant_derivative(x0, It, aux)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pos_pred, cell_pred = model.v(It, t, t)
            L_pos = self.compute_loss(torch.sum((pos_pred - It_dot.pos) ** 2, dim=-1))
            L_cell = self.compute_loss(
                torch.sum((cell_pred - It_dot.cell) ** 2, dim=(-1, -2))
            )
            loss = L_pos + L_cell
        return loss, {"loss": loss, "L_pos": L_pos, "L_cell": L_cell}
