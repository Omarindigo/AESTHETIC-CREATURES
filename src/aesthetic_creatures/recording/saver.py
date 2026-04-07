from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_rollout_npz(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in data.items() if isinstance(v, np.ndarray) or isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str)}
    if "final_info" in serializable:
        del serializable["final_info"]
    np.savez_compressed(output_path, **serializable)


def append_metrics_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    serializable_row = {k: v for k, v in row.items() if not isinstance(v, (dict, list)) or isinstance(v, (list, tuple))}
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serializable_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(serializable_row)
