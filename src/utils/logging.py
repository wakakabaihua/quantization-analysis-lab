"""
Experiment logging utilities.

Provides structured, append-only logging for experiment runs.
Each call to log() appends a record. save() flushes records to both
JSON (for programmatic consumption) and CSV (for quick inspection).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List


class ExperimentLogger:
    """
    Logs experiment records and persists them to JSON and CSV.

    Usage::

        logger = ExperimentLogger("results/raw", "int8_per_tensor_sym")
        logger.log({"model": "mlp", "cosine_similarity": 0.9998, ...})
        logger.save()
    """

    def __init__(self, output_dir: str, experiment_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.records: List[dict] = []

    def log(self, record: dict) -> None:
        """Append a result record, automatically adding a timestamp."""
        self.records.append(
            {"timestamp": time.time(), "experiment": self.experiment_name, **record}
        )

    def save(self) -> None:
        """Flush all records to disk as JSON and CSV."""
        if not self.records:
            return

        json_path = self.output_dir / f"{self.experiment_name}.json"
        with open(json_path, "w") as f:
            json.dump(self.records, f, indent=2)

        csv_path = self.output_dir / f"{self.experiment_name}.csv"
        # Collect all keys from all records to handle heterogeneous dicts
        all_keys = list(dict.fromkeys(k for r in self.records for k in r))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)

    def __repr__(self) -> str:
        return (
            f"ExperimentLogger(name={self.experiment_name!r}, "
            f"records={len(self.records)}, dir={self.output_dir})"
        )
