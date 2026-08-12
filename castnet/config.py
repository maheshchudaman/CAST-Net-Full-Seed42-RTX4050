from pathlib import Path

import yaml

from .model import CASTNetConfig


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_config(values):
    return CASTNetConfig(**values.get("model", {}))

