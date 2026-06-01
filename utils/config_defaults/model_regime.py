from __future__ import annotations

from typing import Any, Dict


FUTURE_CONTEXT_DEFAULTS: Dict[str, Any] = {
    "future_context_hidden_dim": 64,
    "tod_emb_dim": 16,
    "dow_emb_dim": 8,
    "use_future_weekend_context_feature": True,
    "use_future_holiday_context_feature": True,
    "weekend_emb_dim": 4,
    "holiday_emb_dim": 4,
}
