from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def load_config(path: str) -> Dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None:
        raise ValueError("Config path must not be None.")
    with open(resolved, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
    validate_config(config)
    return config

