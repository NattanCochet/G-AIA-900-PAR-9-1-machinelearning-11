"""Append one experiment's result to a shared CSV log.

Every task writes to its own CSV under ``results/`` with the same schema, so
figures and tables can be built directly from these files (project rule: a
split and a timing are mandatory for every reported score).
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

FIELDNAMES = [
    "date",
    "task",
    "seed",
    "config",
    "split",
    "metric",
    "score",
    "duration_seconds",
]


def log_result(
    csv_path: str | Path,
    *,
    task: str,
    seed: int,
    split: str,
    metric: str,
    score: float,
    duration_seconds: float,
    config: dict | None = None,
) -> None:
    """Append one result row to ``csv_path``, creating the file (with header)
    if it doesn't exist yet.

    split: which data the score comes from, e.g. "train", "validation",
        "test" or "cross_validation" -- never "test" for hyperparameter
        tuning.
    config: the run's hyperparameters/settings (seed, lr, batch size, model
        name, ...), stored as a JSON string so the row stays a single CSV
        line.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not csv_path.exists()

    row = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": task,
        "seed": seed,
        "config": json.dumps(config or {}, sort_keys=True),
        "split": split,
        "metric": metric,
        "score": score,
        "duration_seconds": duration_seconds,
    }

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
