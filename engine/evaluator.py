from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from utils.io import iter_batches
from utils.losses import MaskValue, compute_metrics

def maybe_apply_ema(model, ema, fn):
    if ema is None:
        return fn()
    ema.apply_to(model)
    try:
        return fn()
    finally:
        ema.restore(model)


def _cache(value: torch.Tensor, offload: bool) -> torch.Tensor:
    value = value.detach()
    return value.cpu() if offload else value


def _metric_row(
    horizon,
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    mae_mask_value: MaskValue,
    rmse_mask_value: MaskValue,
    mape_mask_value: MaskValue,
    mape_eps: float,
) -> Dict:
    mae, mape, rmse = compute_metrics(
        preds,
        labels,
        mae_mask_value=mae_mask_value,
        rmse_mask_value=rmse_mask_value,
        mape_mask_value=mape_mask_value,
        mape_eps=mape_eps,
    )
    row = {"horizon": horizon, "mae": mae, "rmse": rmse, "mape": mape}
    return row


def evaluate_model(
    model,
    loader,
    scaler,
    device: torch.device,
    *,
    max_batches: Optional[int],
    mae_mask_value: MaskValue,
    rmse_mask_value: MaskValue,
    mape_mask_value: MaskValue,
    mape_eps: float,
    offload_predictions_to_cpu: bool,
) -> list[Dict]:
    model.eval()
    preds_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.no_grad():
        for _, x, y in iter_batches(loader, device, max_batches):
            labels = y[:, :, :, 0].transpose(1, 2)
            preds = model(x, future_data=y)
            preds_list.append(_cache(scaler.inverse_transform(preds), offload_predictions_to_cpu))
            labels_list.append(_cache(scaler.inverse_transform(labels), offload_predictions_to_cpu))

    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    rows = []
    for index in range(preds_all.shape[-1]):
        rows.append(
            _metric_row(
                index + 1,
                preds_all[:, :, index],
                labels_all[:, :, index],
                mae_mask_value=mae_mask_value,
                rmse_mask_value=rmse_mask_value,
                mape_mask_value=mape_mask_value,
                mape_eps=mape_eps,
            )
        )

    avg_row = _metric_row(
        "avg",
        preds_all,
        labels_all,
        mae_mask_value=mae_mask_value,
        rmse_mask_value=rmse_mask_value,
        mape_mask_value=mape_mask_value,
        mape_eps=mape_eps,
    )
    rows.append(avg_row)
    return rows


def extract_horizon_metrics(rows: Iterable[Dict], horizons: Iterable[int], prefix: str) -> Dict:
    by_horizon = {row["horizon"]: row for row in rows if isinstance(row.get("horizon"), int)}
    metrics = {}
    for horizon in horizons:
        row = by_horizon.get(int(horizon))
        if row is None:
            continue
        for metric_name in (
            "mae",
            "rmse",
            "mape",
        ):
            if metric_name in row and row[metric_name] is not None:
                metrics[f"{prefix}_h{horizon}_{metric_name}"] = float(row[metric_name])
    return metrics


def get_average_metric(rows: Iterable[Dict], metric_name: str = "mae") -> float:
    for row in rows:
        if row.get("horizon") == "avg":
            return float(row[metric_name])
    raise ValueError("Evaluation rows do not contain an avg horizon row.")
