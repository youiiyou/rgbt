from __future__ import annotations

import logging
import random

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset

from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer


class BaseDataset:
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self) -> None:
        table = PrettyTable(["subset", "ids", "tracklets", "captions"])
        table.add_row(
            [
                "train",
                len(self.train_id_container),
                len(self.train_annos),
                len(self.train),
            ]
        )
        table.add_row(
            [
                "test",
                len(self.test_id_container),
                len(self.test_annos),
                len(self.test["captions"]),
            ]
        )
        self.logger.info("\n%s", table)


def tokenize(
    caption: str,
    tokenizer: SimpleTokenizer,
    text_length: int = 77,
    truncate: bool = True,
) -> torch.LongTensor:
    start_token = tokenizer.encoder["<|startoftext|>"]
    end_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [start_token] + tokenizer.encode(caption) + [end_token]
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if not truncate:
            raise RuntimeError(
                f"Caption is too long for context length {text_length}: {caption}"
            )
        tokens = tokens[:text_length]
        tokens[-1] = end_token
    result[: len(tokens)] = torch.tensor(tokens)
    return result


def _read_tracklet(frame_paths: list[str]):
    if not isinstance(frame_paths, list) or not frame_paths:
        raise TypeError("Video samples must contain a non-empty frame path list")
    return [read_image(path) for path in frame_paths]


def _apply_transform(images: list[object], transform):
    if transform is None:
        return images

    # Every frame replays the first frame's RNG state so flips/crops/erasing are
    # spatially consistent across the tracklet. The global RNG advances once.
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    transformed = []
    next_torch_state = None
    next_random_state = None
    next_numpy_state = None
    for frame_index, image in enumerate(images):
        if frame_index > 0:
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        transformed.append(transform(image))
        if frame_index == 0:
            next_torch_state = torch.get_rng_state()
            next_random_state = random.getstate()
            next_numpy_state = np.random.get_state()
    torch.set_rng_state(next_torch_state)
    random.setstate(next_random_state)
    np.random.set_state(next_numpy_state)
    return torch.stack(transformed, dim=0)


class ImageTextDataset(Dataset):
    def __init__(
        self,
        dataset,
        transform=None,
        text_length: int = 77,
        truncate: bool = True,
    ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, frame_paths, caption, modality = self.dataset[index]
        images = _apply_transform(_read_tracklet(frame_paths), self.transform)
        caption_ids = tokenize(
            caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return {
            "pids": int(pid),
            "image_ids": int(image_id),
            "images": images,
            "caption_ids": caption_ids,
            "modalities": int(modality),
        }


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, image_modalities, transform=None):
        if not (len(image_pids) == len(img_paths) == len(image_modalities)):
            raise ValueError("Gallery PID/path/modality lengths must match")
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.image_modalities = image_modalities
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        images = _apply_transform(
            _read_tracklet(self.img_paths[index]), self.transform
        )
        return (
            int(self.image_pids[index]),
            int(self.image_modalities[index]),
            images,
        )


class TextDataset(Dataset):
    def __init__(
        self,
        caption_pids,
        captions,
        text_length: int = 77,
        truncate: bool = True,
    ):
        if len(caption_pids) != len(captions):
            raise ValueError("Caption PID/text lengths must match")
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        return int(self.caption_pids[index]), tokenize(
            self.captions[index],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
