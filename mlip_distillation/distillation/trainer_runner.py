import warnings
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, DistributedSampler

from mlip_distillation.dataset.data_utils import pair_atomicdata_list_to_batch
from mlip_distillation.dataset.pair_dataset import AsePairDBDataset
from mlip_distillation.distillation.flow_matching_trainer import FlowMatchingTrainer
from mlip_distillation.distillation.lsd_trainer import LSDTrainer
from mlip_distillation.distributed_utils import DDPWrapper as DDP
from mlip_distillation.models.equiformer_v3.equiformer_v3_flow import EquiformerV3Flow
from mlip_distillation.models.flow_map import FlowMap
from mlip_distillation.models.velocity_field import VelocityField

__TRAINER_RUNNERS__ = {}


def register_trainer_runner(mode: str):
    def wrapper(cls):
        if __TRAINER_RUNNERS__.get(mode, None):
            if __TRAINER_RUNNERS__[mode] != cls:
                warnings.warn(f"Mode {mode} is already registered!", UserWarning)
        __TRAINER_RUNNERS__[mode] = cls
        cls.mode = mode
        return cls

    return wrapper


def get_trainer_runner(config, device, distributed, logger):
    mode = config["mode"]
    assert (
        mode in __TRAINER_RUNNERS__
    ), f"Unknown mode {mode}, available: {sorted(__TRAINER_RUNNERS__)}"
    return __TRAINER_RUNNERS__[mode](config, device, distributed, logger)


class TrainerRunner(ABC):
    """
    Sets up everything needed for a training run (data, model, checkpoint,
    DDP, optimizer, trainer) and runs it. Subclasses choose the model and the
    trainer; device, DDP and seeding are set up by the caller.
    """

    def __init__(self, config, device, distributed, logger):
        self.config = config
        self.device = device
        self.distributed = distributed
        self.logger = logger

    @abstractmethod
    def build_model(self):
        """Returns the (unwrapped) model, on CPU or on self.device."""
        pass

    @abstractmethod
    def build_trainer(self, dataloader, sampler):
        """Returns a Trainer."""
        pass

    def build_dataloader(self):
        dataset = AsePairDBDataset({"src": self.config["data_path"]})
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            loader_kwargs = {"sampler": sampler}
        else:
            sampler = None
            loader_kwargs = {"shuffle": True}

        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            collate_fn=pair_atomicdata_list_to_batch,
            num_workers=self.config["num_workers"],
            pin_memory=(self.device.type == "cuda"),
            **loader_kwargs,
        )
        return dataloader, sampler

    def build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), self.config["lr"])

    def build_backbone(self):
        return EquiformerV3Flow(
            drop_path_rate=self.config["drop_path_rate"],
            attn_weights_drop=self.config["attn_weights_drop"],
        )

    def run(self):
        dataloader, sampler = self.build_dataloader()
        model = self.build_model().to(self.device)

        checkpoint = None
        start_epoch = 0
        if self.config["load"]:
            checkpoint = torch.load(self.config["load"], map_location=self.device)
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1

        if self.distributed:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)

        optimizer = self.build_optimizer(model)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        trainer = self.build_trainer(dataloader, sampler)
        return trainer.train(
            model,
            optimizer,
            self.config["num_epochs"],
            self.device,
            start_epoch,
        )


@register_trainer_runner("train_lsd")
class LSDTrainerRunner(TrainerRunner):
    def build_model(self):
        return FlowMap(self.build_backbone(), self.config["learnable_weights_dim"])

    def build_trainer(self, dataloader, sampler):
        return LSDTrainer(
            dataloader, self.logger, self.config["distillation_eta"], sampler
        )


@register_trainer_runner("train_fm")
class FlowMatchingTrainerRunner(TrainerRunner):
    def build_model(self):
        return VelocityField(self.build_backbone())

    def build_trainer(self, dataloader, sampler):
        return FlowMatchingTrainer(dataloader, self.logger, sampler)
