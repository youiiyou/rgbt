from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    if distributed_rank > 0:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_name = "train_log.txt" if if_train else "test_log.txt"
    mode = "w" if if_train else "a"
    file_handler = logging.FileHandler(
        output_dir / log_name,
        mode=mode,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
