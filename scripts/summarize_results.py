#!/usr/bin/env python3
"""Validate and summarize RGB, IR, and mixed gallery result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prettytable import PrettyTable


GALLERY_MODES = ("rgb", "ir", "mixed")
METRICS = ("R1", "R5", "R10", "mAP", "mINP")
CONSISTENT_FIELDS = (
    "dataset_name",
    "data_root",
    "annotation_source_path",
    "annotation_snapshot_path",
    "annotation_sha256",
    "configured_num_queries",
    "source_commit",
    "source_dirty_at_training_start",
    "checkpoint_path",
    "num_frames",
    "sequence_length",
    "sampling_policy",
    "checkpoint_selection_gallery",
    "train_caption_mode",
    "train_modalities",
    "fake_ir_policy",
    "caption_source",
    "protocol_dir",
    "protocol_sha256",
    "loss_names",
    "seed",
    "total_parameters",
    "trainable_parameters",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def load_results(run_dir: Path) -> dict[str, dict[str, object]]:
    results = {}
    for mode in GALLERY_MODES:
        path = run_dir / f"eval_{mode}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing result file: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("gallery_mode") != mode:
            raise RuntimeError(f"Gallery mode mismatch in {path}")
        results[mode] = result
    return results


def validate_results(results: dict[str, dict[str, object]]) -> None:
    reference = results["mixed"]
    for mode, result in results.items():
        for field in CONSISTENT_FIELDS:
            if result.get(field) != reference.get(field):
                raise RuntimeError(
                    f"{field} differs between mixed and {mode} results"
                )
        if result["num_queries"] != result["configured_num_queries"]:
            raise RuntimeError(f"Query count mismatch in {mode} result")
        if "reverse" not in result:
            raise RuntimeError(f"Reverse retrieval metrics are missing for {mode}")


def rows(results: dict[str, dict[str, object]]):
    for mode in GALLERY_MODES:
        result = results[mode]
        yield ["t2v", mode, result["num_gallery"], *(result[key] for key in METRICS)]
        reverse = result["reverse"]
        yield ["v2t", mode, result["num_queries"], *(reverse[key] for key in METRICS)]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    results = load_results(run_dir)
    validate_results(results)

    table = PrettyTable(["task", "gallery", "candidates", *METRICS])
    all_rows = list(rows(results))
    for row in all_rows:
        table.add_row(row)
    for metric in METRICS:
        table.custom_format[metric] = lambda _field, value: f"{value:.3f}"
    print(table)

    summary = {
        "metadata": {
            key: results["mixed"].get(key) for key in CONSISTENT_FIELDS
        },
        "results": results,
    }
    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / "eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "gallery", "candidates", *METRICS])
        writer.writerows(all_rows)
    print(f"saved: {summary_path}")
    print(f"saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
