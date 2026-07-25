from __future__ import annotations

import torch
import torch.nn as nn

from . import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights


class IRRA(nn.Module):
    """Shared-text IRRA video baseline with uniform frame averaging."""

    SUPPORTED_TASKS = {"sdm", "id"}

    def __init__(self, args, num_classes: int):
        super().__init__()
        self.args = args
        self.num_classes = int(num_classes)
        self.current_task = [
            name.strip() for name in args.loss_names.split("+") if name.strip()
        ]
        unsupported = set(self.current_task) - self.SUPPORTED_TASKS
        if unsupported:
            raise ValueError(
                f"Baseline supports only {sorted(self.SUPPORTED_TASKS)}, got {sorted(unsupported)}"
            )
        if not self.current_task:
            raise ValueError("At least one baseline loss must be enabled")
        if not str(args.pretrain_choice).startswith("ViT"):
            raise ValueError(
                "The video baseline requires a CLIP ViT backbone because it averages "
                "per-frame CLS tokens"
            )

        self.base_model, base_config = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
        )
        self.embed_dim = int(base_config["embed_dim"])
        self.register_buffer(
            "logit_scale",
            torch.tensor(1.0 / float(args.temperature), dtype=torch.float32),
        )

        if "id" in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.zeros_(self.classifier.bias)

        print(f"Training shared-text video baseline with {self.current_task}")

    def _encode_frame_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            if int(self.args.num_frames) != 1:
                raise ValueError(
                    f"Configured {self.args.num_frames} frames but received a 4D image batch"
                )
            tokens = self.base_model.encode_image(images)
            if tokens.ndim != 3:
                raise ValueError(
                    f"Expected ViT image tokens [B,L,D], got {tuple(tokens.shape)}"
                )
            return tokens.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(
                f"Expected images [B,T,C,H,W] or [B,C,H,W], got {tuple(images.shape)}"
            )
        batch_size, num_frames, channels, height, width = images.shape
        if num_frames != int(self.args.num_frames):
            raise ValueError(
                f"Configured {self.args.num_frames} frames but received {num_frames}"
            )
        flat_images = images.reshape(
            batch_size * num_frames, channels, height, width
        )
        tokens = self.base_model.encode_image(flat_images)
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected ViT image tokens [B*T,L,D], got {tuple(tokens.shape)}"
            )
        token_count, embed_dim = tokens.shape[1:]
        return tokens.reshape(
            batch_size, num_frames, token_count, embed_dim
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        frame_tokens = self._encode_frame_tokens(images)
        frame_cls = frame_tokens[:, :, 0, :].float()
        return frame_cls.mean(dim=1)

    def encode_text(self, caption_ids: torch.Tensor) -> torch.Tensor:
        token_features = self.base_model.encode_text(caption_ids)
        batch_indices = torch.arange(
            token_features.shape[0], device=token_features.device
        )
        end_positions = caption_ids.argmax(dim=-1)
        return token_features[batch_indices, end_positions].float()

    def forward(self, batch):
        required = {"images", "caption_ids", "pids", "modalities"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Training batch is missing fields: {sorted(missing)}")
        batch_size = batch["images"].shape[0]
        if any(batch[key].shape[0] != batch_size for key in required - {"images"}):
            raise ValueError("Training batch fields have inconsistent batch sizes")
        if torch.any((batch["modalities"] != 0) & (batch["modalities"] != 1)):
            raise ValueError("Training batch contains an unsupported modality")
        image_features = self.encode_image(batch["images"])
        text_features = self.encode_text(batch["caption_ids"])
        outputs = {"temperature": 1.0 / self.logit_scale}

        if "sdm" in self.current_task:
            outputs["sdm_loss"] = objectives.compute_sdm(
                image_features,
                text_features,
                batch["pids"],
                self.logit_scale,
            )

        if "id" in self.current_task:
            classifier_dtype = self.classifier.weight.dtype
            image_logits = self.classifier(
                image_features.to(classifier_dtype)
            ).float()
            text_logits = self.classifier(
                text_features.to(classifier_dtype)
            ).float()
            outputs["id_loss"] = (
                objectives.compute_id(
                    image_logits,
                    text_logits,
                    batch["pids"],
                )
                * self.args.id_loss_weight
            )
            outputs["img_acc"] = (
                image_logits.argmax(dim=1) == batch["pids"]
            ).float().mean()
            outputs["txt_acc"] = (
                text_logits.argmax(dim=1) == batch["pids"]
            ).float().mean()

        return outputs


def build_model(args, num_classes: int):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model
