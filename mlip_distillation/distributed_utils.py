import os

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed_launch() -> bool:
    """True only when launched via torchrun with more than 1 process."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(use_gpu):
    """
    Call once at the very start of main().
    Returns (device, distributed: bool).
    Handles all three cases: python main.py on GPU, python main.py on CPU,
    and torchrun main.py on N GPUs.
    """
    if not is_distributed_launch():
        device = torch.device("cuda" if use_gpu else "cpu")
        return device, False

    # torchrun path — multiple processes, one per GPU
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    if use_gpu:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return device, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class DDPWrapper(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
