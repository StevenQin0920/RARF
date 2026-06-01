from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from utils.dataset_specs import canonicalize_dataset_name


FEATURE_CHANNELS = (
    "traffic",
    "time_in_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
)
ARTIFACT_MODES = {"series", "split_npz", "both"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, data: Dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def _utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _canonical_dataset_dir(root: Path, dataset_name: str) -> Path:
    return root / canonicalize_dataset_name(dataset_name)
