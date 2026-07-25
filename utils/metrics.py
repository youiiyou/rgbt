from __future__ import annotations

import logging

import torch
from prettytable import PrettyTable

from model.objectives import compute_similarity


def rank(similarity, q_pids, g_pids, max_rank=10):
    if similarity.ndim != 2:
        raise ValueError("Similarity must be a two-dimensional matrix")
    if similarity.shape != (len(q_pids), len(g_pids)):
        raise ValueError("Similarity shape does not match query/gallery IDs")
    if len(g_pids) == 0:
        raise ValueError("Gallery is empty")

    indices = torch.argsort(similarity, dim=1, descending=True)
    predicted_pids = g_pids[indices.cpu()]
    matches = predicted_pids.eq(q_pids.reshape(-1, 1))
    relevant = matches.sum(dim=1)
    if torch.any(relevant == 0):
        missing = int((relevant == 0).sum().item())
        raise ValueError(f"{missing} queries have no positive sample in the gallery")

    effective_rank = min(max_rank, matches.shape[1])
    cmc = matches[:, :effective_rank].cumsum(dim=1)
    cmc[cmc > 1] = 1
    cmc = cmc.float().mean(dim=0) * 100

    cumulative = matches.cumsum(dim=1)
    last_positive_precision = []
    for row_index, match_row in enumerate(matches):
        last_index = match_row.nonzero(as_tuple=False)[-1, 0]
        last_positive_precision.append(
            cumulative[row_index, last_index].float() / float(last_index + 1)
        )
    mean_inp = torch.stack(last_positive_precision).mean() * 100

    positions = torch.arange(
        1, matches.shape[1] + 1, dtype=torch.float32, device=matches.device
    )
    precision_at_rank = cumulative.float() / positions.reshape(1, -1)
    average_precision = (precision_at_rank * matches).sum(dim=1) / relevant
    mean_ap = average_precision.mean() * 100
    return cmc.cpu(), float(mean_ap.item()), float(mean_inp.item()), indices


def _rank_value(cmc, rank_index):
    return float(cmc[min(rank_index - 1, len(cmc) - 1)].item())


class Evaluator:
    def __init__(self, img_loader, txt_loader, gallery_mode="mixed"):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.gallery_mode = gallery_mode
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model.eval()
        device = next(model.parameters()).device
        query_ids, gallery_ids, modalities = [], [], []
        text_features, image_features = [], []

        for pid, caption_ids in self.txt_loader:
            with torch.no_grad():
                features = model.encode_text(caption_ids.to(device))
            query_ids.append(pid.reshape(-1))
            text_features.append(features)

        for pid, modality, images in self.img_loader:
            with torch.no_grad():
                features = model.encode_image(images.to(device))
            gallery_ids.append(pid.reshape(-1))
            modalities.append(modality.reshape(-1))
            image_features.append(features)

        return (
            torch.cat(text_features, dim=0),
            torch.cat(image_features, dim=0),
            torch.cat(query_ids, dim=0),
            torch.cat(gallery_ids, dim=0),
            torch.cat(modalities, dim=0),
        )

    def eval(self, model, include_reverse=True):
        text_features, image_features, query_ids, gallery_ids, modalities = (
            self._compute_embedding(model)
        )
        similarity = compute_similarity(text_features, image_features)
        rgb_count = int((modalities == 0).sum().item())
        ir_count = int((modalities == 1).sum().item())
        if rgb_count + ir_count != len(modalities):
            raise ValueError("Gallery contains an unsupported modality")

        cmc, mean_ap, mean_inp, _ = rank(
            similarity, query_ids, gallery_ids, max_rank=10
        )
        metrics = {
            "task": "text-to-video",
            "gallery_mode": self.gallery_mode,
            "num_queries": int(len(query_ids)),
            "num_gallery": int(len(gallery_ids)),
            "num_rgb_gallery": rgb_count,
            "num_ir_gallery": ir_count,
            "R1": _rank_value(cmc, 1),
            "R5": _rank_value(cmc, 5),
            "R10": _rank_value(cmc, 10),
            "mAP": mean_ap,
            "mINP": mean_inp,
        }

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(
            [
                f"t2v-{self.gallery_mode}",
                metrics["R1"],
                metrics["R5"],
                metrics["R10"],
                metrics["mAP"],
                metrics["mINP"],
            ]
        )
        if include_reverse:
            reverse_cmc, reverse_map, reverse_minp, _ = rank(
                similarity.t(), gallery_ids, query_ids, max_rank=10
            )
            metrics["reverse"] = {
                "task": "video-to-text",
                "R1": _rank_value(reverse_cmc, 1),
                "R5": _rank_value(reverse_cmc, 5),
                "R10": _rank_value(reverse_cmc, 10),
                "mAP": reverse_map,
                "mINP": reverse_minp,
            }
            table.add_row(
                [
                    f"v2t-{self.gallery_mode}",
                    metrics["reverse"]["R1"],
                    metrics["reverse"]["R5"],
                    metrics["reverse"]["R10"],
                    metrics["reverse"]["mAP"],
                    metrics["reverse"]["mINP"],
                ]
            )

        for column in ("R1", "R5", "R10", "mAP", "mINP"):
            table.custom_format[column] = lambda _field, value: f"{value:.3f}"
        self.logger.info(
            "Gallery=%s queries=%d gallery=%d (rgb=%d ir=%d)\n%s",
            self.gallery_mode,
            len(query_ids),
            len(gallery_ids),
            rgb_count,
            ir_count,
            table,
        )
        return metrics
