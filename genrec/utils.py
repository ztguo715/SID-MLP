import ast
import datetime
import hashlib
import html
import importlib
import logging
import os
import re
from logging import getLogger
from typing import Optional, Union

import yaml
from accelerate.utils import set_seed

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel


def init_seed(seed, reproducibility):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = not reproducibility
    torch.backends.cudnn.deterministic = bool(reproducibility)


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M")


def get_file_name(config: dict, suffix: str = ""):
    public_config = {
        key: value
        for key, value in config.items()
        if key not in {"accelerator", "device"}
    }
    config_str = repr(sorted(public_config.items()))
    md5 = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]
    run_id = str(config.get("run_id", "genrec"))
    run_time = str(config.get("run_local_time", get_local_time()))
    return f"{run_id}-{run_time}-{md5}{suffix}"


def init_logger(config: dict):
    log_root = config["log_dir"]
    log_dir = os.path.join(log_root, config["dataset"], config["model"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, get_file_name(config, suffix=".log"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)s  %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    if not config["accelerator"].is_main_process:
        try:
            from datasets.utils.logging import disable_progress_bar

            disable_progress_bar()
        except Exception:
            pass


def log(message, accelerator, logger, level="info"):
    if not accelerator.is_main_process:
        return
    getattr(logger, level)(message)


def get_tokenizer(model_name: str):
    if model_name != "TIGER":
        raise ValueError(f'Model "{model_name}" is not included in this TIGER-only copy.')
    return getattr(importlib.import_module("genrec.models.TIGER.tokenizer"), "TIGERTokenizer")


def get_model(model_name: Union[str, AbstractModel]):
    if isinstance(model_name, AbstractModel):
        return model_name
    if model_name != "TIGER":
        raise ValueError(f'Model "{model_name}" is not included in this TIGER-only copy.')
    return getattr(importlib.import_module("genrec.models"), "TIGER")


def get_dataset(dataset_name: Union[str, AbstractDataset]):
    if isinstance(dataset_name, AbstractDataset):
        return dataset_name
    if dataset_name != "AmazonReviews2023":
        raise ValueError(f'Dataset "{dataset_name}" is not included in this TIGER-only copy.')
    return getattr(importlib.import_module("genrec.datasets"), "AmazonReviews2023")


def get_trainer(model_name: Union[str, AbstractModel]):
    from genrec.trainer import Trainer

    return Trainer


def get_pipeline(model_name: Union[str, AbstractModel]):
    from genrec.pipeline import Pipeline

    return Pipeline


def get_total_steps(config, train_dataloader):
    if config["steps"] is not None:
        return int(config["steps"])
    return len(train_dataloader) * int(config["epochs"])


def _coerce_value(value):
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def convert_config_dict(config: dict) -> dict:
    return {key: _coerce_value(value) for key, value in config.items()}


def _load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_config(
    model_name: Union[str, AbstractModel],
    dataset_name: Union[str, AbstractDataset],
    config_file: Union[str, list[str], None] = None,
    config_dict: Optional[dict] = None,
) -> dict:
    final_config = {}
    logger = getLogger()
    package_dir = os.path.dirname(os.path.realpath(__file__))
    config_files = [os.path.join(package_dir, "default.yaml")]

    if isinstance(dataset_name, str):
        config_files.append(os.path.join(package_dir, "datasets", dataset_name, "config.yaml"))
        final_config["dataset"] = dataset_name
    else:
        logger.info("Custom dataset config must be supplied by config_file or config_dict.")
        final_config["dataset"] = dataset_name.__class__.__name__

    if isinstance(model_name, str):
        config_files.append(os.path.join(package_dir, "models", model_name, "config.yaml"))
        final_config["model"] = model_name
    else:
        logger.info("Custom model config must be supplied by config_file or config_dict.")
        final_config["model"] = model_name.__class__.__name__

    if config_file:
        config_files.extend([config_file] if isinstance(config_file, str) else config_file)

    for path in config_files:
        final_config.update(_load_yaml(path))

    if config_dict:
        final_config.update(config_dict)

    final_config["run_local_time"] = get_local_time()
    return convert_config_dict(final_config)


def parse_command_line_args(unparsed: list[str]) -> dict:
    args = {}
    idx = 0
    while idx < len(unparsed):
        text = unparsed[idx]
        if not text.startswith("--"):
            raise ValueError(f"Unexpected argument: {text}")
        if "=" in text:
            key, value = text[2:].split("=", 1)
            idx += 1
        else:
            if idx + 1 >= len(unparsed) or unparsed[idx + 1].startswith("--"):
                raise ValueError(f"Missing value for argument: {text}")
            key, value = text[2:], unparsed[idx + 1]
            idx += 2
        args[key] = _coerce_value(value)
    return args


def list_to_str(value: Union[list, str], remove_blank=False) -> str:
    text = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return text.replace(" ", "") if remove_blank else text


def clean_text(raw_text: str) -> str:
    text = html.unescape(list_to_str(raw_text)).strip()
    text = re.sub(r"u['\"]", "", text)
    text = re.sub(r"^['\"]|['\"]$", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    return re.sub(r"[^\x00-\x7F]", " ", text)


def init_device():
    import torch

    use_ddp = bool(os.environ.get("WORLD_SIZE"))
    if torch.cuda.is_available():
        return torch.device("cuda"), use_ddp
    return torch.device("cpu"), use_ddp


def config_for_log(config: dict) -> dict:
    logged = config.copy()
    logged.pop("device", None)
    logged.pop("accelerator", None)
    for key, value in logged.items():
        if isinstance(value, list):
            logged[key] = str(value)
    return logged
