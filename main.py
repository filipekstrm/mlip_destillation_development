import numpy as np
import torch

from mlip_distillation.common.args_and_config import get_config
from mlip_distillation.common.loggers import get_logger
from mlip_distillation.distillation.trainer_runner import get_trainer_runner
from mlip_distillation.distributed_utils import cleanup_ddp, setup_ddp


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config):
    seed_all(config["seed"])

    device, distributed = setup_ddp(config["cuda"])
    if device.type == "cuda":
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger = get_logger(config)

    runner = get_trainer_runner(config, device, distributed, logger)
    runner.run()

    cleanup_ddp()


if __name__ == "__main__":
    main(get_config())
