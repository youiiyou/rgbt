from __future__ import annotations

import torch
import torch.nn.functional as functional


def compute_similarity(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
) -> torch.Tensor:
    text_normalized = functional.normalize(text_features.float(), dim=-1)
    image_normalized = functional.normalize(image_features.float(), dim=-1)
    return text_normalized @ image_normalized.t()


def compute_sdm(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pids: torch.Tensor,
    logit_scale: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Similarity Distribution Matching from IRRA."""
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError("SDM requires paired image/text batches")
    batch_size = image_features.shape[0]
    pids = pids.reshape(batch_size, 1)
    labels = (pids == pids.t()).float()
    target_distribution = labels / labels.sum(dim=1, keepdim=True)

    text_to_image = logit_scale * compute_similarity(text_features, image_features)
    image_to_text = text_to_image.t()

    image_to_text_prob = functional.softmax(image_to_text, dim=1)
    image_to_text_loss = image_to_text_prob * (
        functional.log_softmax(image_to_text, dim=1)
        - torch.log(target_distribution + epsilon)
    )
    text_to_image_prob = functional.softmax(text_to_image, dim=1)
    text_to_image_loss = text_to_image_prob * (
        functional.log_softmax(text_to_image, dim=1)
        - torch.log(target_distribution + epsilon)
    )
    return image_to_text_loss.sum(dim=1).mean() + text_to_image_loss.sum(dim=1).mean()


def compute_id(
    image_logits: torch.Tensor,
    text_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    image_loss = functional.cross_entropy(image_logits, labels)
    text_loss = functional.cross_entropy(text_logits, labels)
    return 0.5 * (image_loss + text_loss)
