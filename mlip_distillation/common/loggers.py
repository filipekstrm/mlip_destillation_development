import os
import shutil
import sys
import warnings

import torch
import wandb

from mlip_distillation.common.constants import (
    BEST_PARAMS_FILE,
    PARAMS_FILE,
    WANDB_PROJECT,
)
from mlip_distillation.distributed_utils import is_main_process

__LOGGERS__ = {}


def register_logger(name: str):
    def wrapper(cls):
        if __LOGGERS__.get(name, None):
            if __LOGGERS__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __LOGGERS__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_logger(config):
    """
    Logger given by config["logger"] on the main process, a NoOpLogger on all
    other ranks, so only the main process creates directories, logs and saves.
    """
    if not is_main_process():
        return NoOpLogger(config)
    name = config["logger"]
    assert (
        name in __LOGGERS__
    ), f"Unknown logger {name}, available: {sorted(__LOGGERS__)}"
    return __LOGGERS__[name](config)


class BaseLogger:
    def __init__(self, config):
        if is_main_process():
            print(
                f"Setting up logger, using {self.__class__.__qualname__}",
                file=sys.stdout,
            )

    def log(self, data_to_log, step):
        raise NotImplementedError

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        raise NotImplementedError

    def update_summary(self, log):
        raise NotImplementedError


@register_logger("none")
@register_logger("noop")
class NoOpLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

    def log(self, data_to_log, step):
        return

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        return

    def update_summary(self, log):
        return


def create_checkpoint_directory(run_name, load_path: str):
    os.makedirs("checkpoints", exist_ok=True)
    if load_path:
        split_path = load_path.split("/")
        assert (
            "checkpoints" in split_path
        ), "Ensure that load parameters is in checkpoints directory."
        # # Use if we want to old directory
        # # Get up until directory only
        # checkpoint_dir = split_path[0:-1]
        # TODO: If we want this style, rewrite and put the assert below here and check for "if not load_path: ..."
        checkpoint_dir = os.path.join("checkpoints", run_name)
        os.mkdir(checkpoint_dir)
    else:
        checkpoint_dir = os.path.join("checkpoints", run_name)
        assert not os.path.isdir(checkpoint_dir), "Checkpoint directory already exists"
        os.mkdir(checkpoint_dir)
    return checkpoint_dir


def create_log_directory(run_name, load_path: str):
    os.makedirs("logs", exist_ok=True)
    if load_path:
        split_path = load_path.split("/")
        assert (
            "checkpoints" in split_path
        ), "Ensure that load parameters is in checkpoints directory."
        # # Use if we want to old directory
        for idx, element in enumerate(split_path):
            if element == "checkpoints":
                # Get next element which should be checkpoint dir.
                # Get up until last directory only
                prev_log_dir = os.path.join("logs", *split_path[idx + 1 : -1])
        # TODO: If we want this style, rewrite and put the assert below here and check for "if not load_path: ..."
        log_dir = os.path.join("logs", run_name)
        # Create new logdir
        os.mkdir(log_dir)

        # Copy previous log files to new dir.
        for log_file in os.listdir(prev_log_dir):
            shutil.copy(os.path.join(prev_log_dir, log_file), log_dir)

    else:
        log_dir = os.path.join("logs", run_name)
        assert not os.path.isdir(log_dir), "Log directory already exists"
        os.mkdir(log_dir)
    return log_dir


@register_logger("model_only")
class ModelOnlyLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in checkpoints/
        self.checkpoint_dir = create_checkpoint_directory(
            config["run_name"], config["load"]
        )

    def update_summary(self, log):
        pass

    def log(self, data_to_log, step):
        pass

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        if best:
            name = BEST_PARAMS_FILE
        elif epoch is not None:
            name, ext = os.path.splitext(PARAMS_FILE)
            name = f"{name}_{epoch}{ext}"
        else:
            name = PARAMS_FILE
        torch.save(dict_to_save, os.path.join(self.checkpoint_dir, name))


@register_logger("local_only")
class LocalOnlyLogger(ModelOnlyLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in logs/ where all logging is saved
        self.log_dir = create_log_directory(config["run_name"], config["load"])

        # create log file
        self.log_file = os.path.join(self.log_dir, "full_log.out")

    def update_summary(self, log):
        pass

    def log(self, data_to_log, step):
        full_dict = {"Step": step}
        full_dict.update(data_to_log)
        with open(self.log_file, "a") as log_file:
            log_file.write(f"{full_dict}\n")


@register_logger("wandb")
class WandBLogger(ModelOnlyLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in logs/ where all logging is saved
        self.log_dir = create_log_directory(config["run_name"], config["load"])

        wandb.init(
            project=WANDB_PROJECT,
            config=config,
            name=config["run_name"],
            dir=self.log_dir,
        )

    def log(self, data_to_log, step):
        wandb.log(data_to_log, step=step)

    def update_summary(self, log):
        for key, value in log.items():
            wandb.run.summary[key] = value
