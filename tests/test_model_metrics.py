import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from model.build import IRRA
from model.objectives import compute_sdm
from solver.lr_scheduler import LRSchedulerWithWarmup
from utils.metrics import Evaluator, rank


class ObjectiveTests(unittest.TestCase):
    def test_sdm_is_finite_and_backpropagates(self):
        torch.manual_seed(5)
        image_features = torch.randn(6, 8, requires_grad=True)
        text_features = torch.randn(6, 8, requires_grad=True)
        pids = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = compute_sdm(
            image_features,
            text_features,
            pids,
            torch.tensor(50.0),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(image_features.grad).all())
        self.assertTrue(torch.isfinite(text_features.grad).all())


class ModelPoolingTests(unittest.TestCase):
    def test_sequence_encoder_averages_per_frame_cls_features(self):
        class FakeClip(nn.Module):
            def encode_image(self, images):
                cls = images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 4)
                patch = torch.zeros_like(cls)
                return torch.stack((cls, patch), dim=1)

        model = IRRA.__new__(IRRA)
        nn.Module.__init__(model)
        model.args = SimpleNamespace(num_frames=2)
        model.base_model = FakeClip()
        images = torch.stack(
            (
                torch.stack((torch.ones(3, 2, 2), torch.full((3, 2, 2), 3.0))),
                torch.stack((torch.full((3, 2, 2), 2.0), torch.full((3, 2, 2), 6.0))),
            )
        )
        features = model.encode_image(images)
        expected = torch.tensor([[2.0] * 4, [4.0] * 4])
        self.assertTrue(torch.equal(features, expected))

    def test_sequence_encoder_rejects_wrong_frame_count(self):
        class FakeClip(nn.Module):
            def encode_image(self, images):
                return torch.zeros(images.shape[0], 2, 4)

        model = IRRA.__new__(IRRA)
        nn.Module.__init__(model)
        model.args = SimpleNamespace(num_frames=6)
        model.base_model = FakeClip()
        with self.assertRaisesRegex(ValueError, "Configured 6 frames"):
            model.encode_image(torch.zeros(2, 5, 3, 2, 2))


class MetricTests(unittest.TestCase):
    def test_rank_rejects_queries_without_gallery_positive(self):
        with self.assertRaisesRegex(ValueError, "no positive"):
            rank(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([3]),
                torch.tensor([1, 2]),
            )

    def test_evaluator_reports_both_directions_and_modality_counts(self):
        class IdentityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def encode_text(self, values):
                return values.float()

            def encode_image(self, values):
                return values.float()

        text_loader = [
            (
                torch.tensor([0, 1]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        image_loader = [
            (
                torch.tensor([0, 1, 0, 1]),
                torch.tensor([0, 0, 1, 1]),
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
                ),
            )
        ]
        metrics = Evaluator(image_loader, text_loader, "mixed").eval(
            IdentityModel(), include_reverse=True
        )
        self.assertEqual(metrics["R1"], 100.0)
        self.assertEqual(metrics["reverse"]["R1"], 100.0)
        self.assertEqual(metrics["num_rgb_gallery"], 2)
        self.assertEqual(metrics["num_ir_gallery"], 2)


class SchedulerTests(unittest.TestCase):
    def test_one_epoch_smoke_schedule_remains_finite(self):
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1e-3)
        scheduler = LRSchedulerWithWarmup(
            optimizer,
            milestones=(20, 50),
            warmup_epochs=1,
            total_epochs=1,
            mode="cosine",
        )
        optimizer.step()
        scheduler.step()
        self.assertTrue(torch.isfinite(torch.tensor(scheduler.get_last_lr())).all())


if __name__ == "__main__":
    unittest.main()
