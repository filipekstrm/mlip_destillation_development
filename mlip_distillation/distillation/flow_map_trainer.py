import torch

from mlip_distillation.distillation.trainer import Trainer


class FlowMapTrainer(Trainer):
    def sample_s_t(self, batch_size, device):
        s = torch.rand((batch_size, 1), device=device)
        t = s + torch.rand_like(s) * (1 - s)
        return s, t

    def compute_loss(
        self,
        weight,
        sq_norm,
    ):
        return torch.mean(torch.exp(-weight) * sq_norm + weight)
