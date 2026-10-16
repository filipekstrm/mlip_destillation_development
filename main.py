import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from mlip_distillation.common.args_and_config import get_config
from mlip_distillation.common.loggers import get_logger
from mlip_distillation.dataset.data_utils import pair_atomicdata_list_to_batch
from mlip_distillation.dataset.pair_dataset import AsePairDBDataset
from mlip_distillation.distillation.lsd_trainer import LSDTrainer
from mlip_distillation.distributed_utils import DDPWrapper as DDP
from mlip_distillation.distributed_utils import cleanup_ddp, setup_ddp
from mlip_distillation.models.equiformer_v3.equiformer_v3_flowmap import (
    EquiformerV3FlowMap,
)
from mlip_distillation.models.flow_map import FlowMap


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

    # Created before config["device"] is set, so the config stays serializable
    logger = get_logger(config)

    config["device"] = device
    # Dataloader
    dataset = AsePairDBDataset({"src": config["data_path"]})
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader_kwargs = {"sampler": sampler}
    else:
        sampler = None
        loader_kwargs = {"shuffle": True}

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=pair_atomicdata_list_to_batch,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        **loader_kwargs
    )

    # Models
    v_model = EquiformerV3FlowMap(
        drop_path_rate=config["drop_path_rate"],
        attn_weights_drop=config["attn_weights_drop"],
    )
    flow_map = FlowMap(v_model, config["learnable_weights_dim"]).to(config["device"])

    checkpoint = None
    start_epoch = 0
    if config["load"]:
        checkpoint = torch.load(config["load"], map_location=device)
        flow_map.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1

    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        flow_map = DDP(flow_map, device_ids=device_ids)

    optimizer = torch.optim.AdamW(flow_map.parameters(), config["lr"])
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Trainer
    trainer = LSDTrainer(dataloader, logger, config["distillation_eta"], sampler)

    # start training
    flow_map = trainer.train(
        flow_map,
        optimizer,
        config["num_epochs"],
        config["device"],
        start_epoch,
    )

    cleanup_ddp()


if __name__ == "__main__":
    main(get_config())
