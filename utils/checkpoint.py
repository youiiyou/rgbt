from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _strip_module_prefix(state_dict):
    if not state_dict or not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state_dict.items()
    )


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=True,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_to_disk = bool(save_to_disk)
        self.logger = logger or logging.getLogger("IRRA.checkpoint")

    def save(self, name, **metadata) -> None:
        if self.save_dir is None or not self.save_to_disk:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {"model": _unwrap_model(self.model).state_dict()}
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        checkpoint.update(metadata)
        destination = self.save_dir / f"{name}.pth"
        self.logger.info("Saving checkpoint to %s", destination)
        torch.save(checkpoint, destination)

    def load(self, checkpoint_path=None) -> dict[str, object]:
        if not checkpoint_path:
            raise FileNotFoundError("No checkpoint path was provided")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        self.logger.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format: {path}")
        self._load_model(checkpoint["model"])
        return checkpoint

    def resume(self, checkpoint_path=None) -> dict[str, object]:
        checkpoint = self.load(checkpoint_path)
        if self.optimizer is not None:
            if "optimizer" not in checkpoint:
                raise RuntimeError("Resume checkpoint has no optimizer state")
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            if "scheduler" not in checkpoint:
                raise RuntimeError("Resume checkpoint has no scheduler state")
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint

    def _load_model(self, state_dict) -> None:
        normalized = _strip_module_prefix(state_dict)
        incompatible = _unwrap_model(self.model).load_state_dict(
            normalized,
            strict=False,
        )
        allowed_historical_missing = {"logit_scale"}
        missing = set(incompatible.missing_keys) - allowed_historical_missing
        unexpected = set(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint/model keys differ; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
