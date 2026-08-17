import argparse
import json
import os
import sys
from datetime import datetime

import torch


def get_parser():
    """
    Function to obtain the config-dictionary with all parameters to be used in training/testing
    :return: config - dictionary with all parameters to be used
    """
    parser = argparse.ArgumentParser(
        description="Generation of Wyckoff positions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # If config file should be used
    parser.add_argument(
        "--config", type=str, help="Config file to read run config from"
    )

    # General
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )

    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Whether to disable cuda, even if available",
    )

    # Model
    parser.add_argument(
        "--learnable_weights_dim",
        type=int,
        default=None,
        help="Dimension for learnable weights model, None means no learnable weights",
    )

    # Training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Seed for random number generator",
    )

    parser.add_argument(
        "--distillation_eta",
        type=float,
        default=0.75,
        help="Fraction of batch used for diagonal loss",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    return parser


def get_included_config(file_path):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)
        if "include" in config:
            included_config = get_included_config(config["include"])
            assert "include" not in included_config
            included_config.update(config)
            del included_config["include"]
            config = included_config.copy()
    return config


def get_config():
    # First parse, and include default values
    default_parser = get_parser()
    default_args = default_parser.parse_args()
    config = vars(default_args)

    # Now update with arguments from config file
    if default_args.config:
        assert os.path.exists(default_args.config), f"No config file: {default_args}"
        with open(default_args.config) as json_file:
            config_from_file = json.load(json_file)
        if (
            "include" in config_from_file
        ):  # ability to include another json to avoid having to specify multiple things
            included_config = get_included_config(config_from_file["include"])
            included_config.update(config_from_file)
            config_from_file = included_config.copy()
            del config_from_file["include"]
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(
            ["Unknown option in config file: {}".format(opt) for opt in unknown_options]
        )
        assert not unknown_options, unknown_error
        config.update(config_from_file)

    # Now parse again, but without any default values
    command_line_parser = argparse.ArgumentParser(
        parents=[default_parser], add_help=False
    )
    command_line_parser.set_defaults(**{key: None for key in config.keys()})
    command_line_args = command_line_parser.parse_args()

    # now overwrite values in config with those from command line
    flags_from_command_line = []
    config_from_command_line = vars(command_line_args)
    for key, value in config_from_command_line.items():
        if value is not None:
            if key == "config":
                value = str(value)
                value = os.path.splitext(value)[0]
                value = "-".join(os.path.normpath(value).split(os.path.sep))
            if key != "logger":
                if key == "load":
                    # Reduce loadstring to date time only for reference.
                    split_value = value.split("/")
                    assert (
                        "checkpoints" in split_value
                    ), "Ensure that load parameters is in checkpoints directory."
                    for idx, element in enumerate(split_value):
                        if element == "checkpoints":
                            #
                            prev_run_name = split_value[idx + 1]
                            string_value = prev_run_name.split("_")[0]
                else:
                    string_value = value
                flags_from_command_line.append(
                    f"{key}={string_value}" if key != "config" else str(string_value)
                )
            config[key] = value

    config["cuda"] = not config["disable_cuda"] and torch.cuda.is_available()
    if "SLURM_JOB_ID" in os.environ:
        config["job_id"] = os.environ.get("SLURM_JOB_ID")
    else:
        config["job_id"] = None
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    config["run_name"] = "_".join([dt_string] + flags_from_command_line)
    # if not config["mode"].startswith("train") or config["debug"]:
    #     config["logger"] = "none"
    #     print(
    #         "Not running any training/in debug mode, therefore using no-op logger",
    #         file=sys.stdout,
    #     )
    return config


if __name__ == "__main__":
    print(get_config(), file=sys.stdout)
