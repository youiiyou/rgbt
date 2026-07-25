import copy
import unittest

from scripts.summarize_results import CONSISTENT_FIELDS, rows, validate_results


def make_result(mode: str) -> dict[str, object]:
    result = {field: f"value-{field}" for field in CONSISTENT_FIELDS}
    result.update(
        {
            "configured_num_queries": 2,
            "num_queries": 2,
            "gallery_mode": mode,
            "num_gallery": 4,
            "R1": 1.0,
            "R5": 2.0,
            "R10": 3.0,
            "mAP": 4.0,
            "mINP": 5.0,
            "reverse": {
                "R1": 1.0,
                "R5": 2.0,
                "R10": 3.0,
                "mAP": 4.0,
                "mINP": 5.0,
            },
        }
    )
    return result


class ResultSummaryTests(unittest.TestCase):
    def test_three_gallery_results_require_identical_run_metadata(self):
        results = {mode: make_result(mode) for mode in ("rgb", "ir", "mixed")}
        validate_results(results)
        changed = copy.deepcopy(results)
        changed["ir"]["seed"] = "different"
        with self.assertRaisesRegex(RuntimeError, "seed differs"):
            validate_results(changed)

    def test_reverse_metrics_are_required(self):
        results = {mode: make_result(mode) for mode in ("rgb", "ir", "mixed")}
        del results["rgb"]["reverse"]
        with self.assertRaisesRegex(RuntimeError, "Reverse retrieval"):
            validate_results(results)

    def test_reverse_rows_report_caption_candidate_count(self):
        results = {mode: make_result(mode) for mode in ("rgb", "ir", "mixed")}

        summary_rows = list(rows(results))

        self.assertEqual(summary_rows[0][2], 4)
        self.assertEqual(summary_rows[1][2], 2)


if __name__ == "__main__":
    unittest.main()
