from __future__ import annotations

import json
from pathlib import Path

import yaml
from easydict import EasyDict
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader with the tuple tag emitted by historical IRRA configs."""


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


def read_image(img_path):
    path = Path(img_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def mkdir_if_missing(directory):
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_train_configs(path, args):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configs.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, default_flow_style=False, sort_keys=True)


def load_train_configs(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return EasyDict(yaml.load(handle, Loader=ConfigLoader))
