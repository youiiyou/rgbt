import random
import unittest
from unittest import mock

import numpy as np
import torch

from datasets.bases import _apply_transform
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from datasets.video_utils import (
    DatasetContractError,
    build_eval_dict,
    natural_key,
    uniform_sample_frames,
)


class VideoSamplingTests(unittest.TestCase):
    def test_natural_sort_orders_numeric_components(self):
        values = ["D10", "D2", "D1"]
        self.assertEqual(sorted(values, key=natural_key), ["D1", "D2", "D10"])

    def test_uniform_segment_midpoints_for_long_tracklet(self):
        frames = [str(index) for index in range(10)]
        self.assertEqual(
            uniform_sample_frames(frames, 6),
            ["0", "1", "3", "5", "6", "8"],
        )

    def test_short_tracklet_repeats_at_evenly_spaced_positions(self):
        self.assertEqual(
            uniform_sample_frames(["a", "b", "c"], 6),
            ["a", "a", "b", "b", "c", "c"],
        )

    def test_evaluation_split_requires_a_positive_for_every_query(self):
        gallery = [{"pid": 0, "img_paths": ["frame"], "modality": 0}]
        queries = [{"pid": 1, "caption": "missing"}]
        with self.assertRaisesRegex(DatasetContractError, "missing positive"):
            build_eval_dict(gallery, queries)

    def test_tracklet_transform_replays_all_supported_rng_states(self):
        class RandomStamp:
            def __call__(self, _image):
                return torch.tensor(
                    [random.random(), float(torch.rand(())), float(np.random.rand())]
                )

        random.seed(17)
        torch.manual_seed(17)
        np.random.seed(17)
        transformed = _apply_transform(
            [object(), object(), object()], RandomStamp()
        )
        self.assertTrue(torch.equal(transformed[0], transformed[1]))
        self.assertTrue(torch.equal(transformed[0], transformed[2]))

        next_values = (random.random(), float(torch.rand(())), float(np.random.rand()))
        random.seed(17)
        torch.manual_seed(17)
        np.random.seed(17)
        RandomStamp()(object())
        expected_next = (
            random.random(),
            float(torch.rand(())),
            float(np.random.rand()),
        )
        self.assertEqual(next_values, expected_next)


class IdentitySamplerTests(unittest.TestCase):
    def test_sampler_accepts_five_field_video_samples(self):
        samples = [
            (pid, image_id, ["frame.jpg"], "caption", image_id % 2)
            for pid in range(3)
            for image_id in range(2)
        ]
        random.seed(3)
        np.random.seed(3)
        sampler = RandomIdentitySampler(samples, batch_size=4, num_instances=2)
        indices = list(iter(sampler))
        self.assertEqual(len(indices) % 4, 0)
        for start in range(0, len(indices), 4):
            pids = [samples[index][0] for index in indices[start : start + 4]]
            self.assertEqual(sorted(pids.count(pid) for pid in set(pids)), [2, 2])

    def test_sampler_rejects_incompatible_batch_shape(self):
        samples = [(0, 0, ["frame.jpg"], "caption", 0)]
        with self.assertRaisesRegex(ValueError, "divisible"):
            RandomIdentitySampler(samples, batch_size=3, num_instances=2)

    def test_distributed_sampler_splits_the_same_global_identity_batches(self):
        samples = [
            (pid, instance, ["frame.jpg"], "caption", instance % 2)
            for pid in range(8)
            for instance in range(2)
        ]

        def build_for_rank(rank):
            with (
                mock.patch("datasets.sampler_ddp.dist.is_available", return_value=True),
                mock.patch("datasets.sampler_ddp.dist.is_initialized", return_value=True),
                mock.patch("datasets.sampler_ddp.dist.get_world_size", return_value=2),
                mock.patch("datasets.sampler_ddp.dist.get_rank", return_value=rank),
            ):
                sampler = RandomIdentitySampler_DDP(
                    samples,
                    global_batch_size=8,
                    num_instances=2,
                    seed=11,
                )
                sampler.set_epoch(3)
                return list(iter(sampler))

        rank_zero = build_for_rank(0)
        rank_one = build_for_rank(1)
        self.assertEqual(len(rank_zero), len(rank_one))
        for start in range(0, len(rank_zero), 4):
            combined = rank_zero[start : start + 4] + rank_one[start : start + 4]
            self.assertEqual(len(combined), len(set(combined)))
            for local in (rank_zero[start : start + 4], rank_one[start : start + 4]):
                pids = [samples[index][0] for index in local]
                self.assertEqual(sorted(pids.count(pid) for pid in set(pids)), [2, 2])


if __name__ == "__main__":
    unittest.main()
