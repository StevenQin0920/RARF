from __future__ import annotations

from typing import Any, Dict

CONFIG_SCHEMA_VERSION = "rarf_slim_v1"

METHOD_DEFAULTS: Dict[str, Any] = {
    "name": "RARF",
    "full_name": "Regime-Anchored Residual Forecasting",
    "title": "Rethinking Spatio-Temporal Sequence Forecasting via Regime-Anchored Residual Dynamics",
    "formula": "Y_hat = A0_future + R_corr",
}
