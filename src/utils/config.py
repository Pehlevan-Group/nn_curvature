"""
parse config
"""

# load packages
import os
import toml
from typing import Dict
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.parent.parent.absolute()


def load_config(tag="simulation") -> Dict[str, str]:
    """load toml file"""
    config_file_path = os.path.join(PACKAGE_DIR, "config.toml")

    config = {}
    toml_dict = toml.load(config_file_path)
    config.update(toml_dict[tag])

    assert "data_dir" in config, "Did not find data_dir"
    assert "model_dir" in config, "Did not find model_dir"
    assert "result_dir" in config, "Did not find result_dir"

    for _, path in config.items():
        if not os.path.exists(path):
            os.mkdir(path)

    return config


def fileid(network_type, args) -> str:
    """consistent fileid upon saving"""
    return "{}_w{}_nl{}_lr{}_wd{}_mom{}_{}_epochs{}_seed{}".format(
        network_type,
        args.w,
        args.nl,
        args.lr,
        args.weight_decay,
        args.momentum,
        args.data,
        args.epochs,
        args.seed,
    )
