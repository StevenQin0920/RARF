from __future__ import annotations

import torch


MaskValue = float | None


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw_mask = mask.float()
    mask = raw_mask / torch.clamp(raw_mask.mean(), min=1e-6)
    loss = loss * mask
    invalid = raw_mask <= 0
    loss = torch.where(invalid & ~torch.isfinite(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def valid_value_mask(labels: torch.Tensor, mask_value: MaskValue = None) -> torch.Tensor:
    valid = torch.isfinite(labels)
    if mask_value is not None:
        valid = valid & (labels != float(mask_value))
    return valid


def _mse(preds: torch.Tensor, labels: torch.Tensor, mask_value: MaskValue = None) -> torch.Tensor:
    loss = (preds - labels) ** 2
    return _masked_mean(loss, valid_value_mask(labels, mask_value))


def rmse(preds: torch.Tensor, labels: torch.Tensor, mask_value: MaskValue = None) -> torch.Tensor:
    return torch.sqrt(_mse(preds=preds, labels=labels, mask_value=mask_value))


def mae_loss(preds: torch.Tensor, labels: torch.Tensor, mask_value: MaskValue = None) -> torch.Tensor:
    loss = torch.abs(preds - labels)
    return _masked_mean(loss, valid_value_mask(labels, mask_value))


def fft_magnitude_loss(preds: torch.Tensor, labels: torch.Tensor, mask_value: MaskValue = None) -> torch.Tensor:
    valid = valid_value_mask(labels, mask_value)
    preds = torch.where(valid, preds, torch.zeros_like(preds)).float()
    labels = torch.where(valid, labels, torch.zeros_like(labels)).float()
    pred_fft = torch.fft.rfft(preds, dim=-1, norm="ortho")
    label_fft = torch.fft.rfft(labels, dim=-1, norm="ortho")
    return torch.abs(torch.abs(pred_fft) - torch.abs(label_fft)).mean()


def mape(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask_value: MaskValue = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    valid = valid_value_mask(labels, mask_value) & (torch.abs(labels) > float(eps))
    loss = torch.abs(preds - labels) / torch.clamp(torch.abs(labels), min=eps)
    return _masked_mean(loss, valid)


def compute_metrics(
    pred: torch.Tensor,
    real: torch.Tensor,
    *,
    mae_mask_value: MaskValue = None,
    rmse_mask_value: MaskValue = None,
    mape_mask_value: MaskValue = None,
    mape_eps: float = 1e-5,
) -> tuple[float, float, float]:
    mae_value = mae_loss(pred, real, mae_mask_value).item()
    mape_value = mape(pred, real, mape_mask_value, eps=mape_eps).item()
    rmse_value = rmse(pred, real, rmse_mask_value).item()
    return mae_value, mape_value, rmse_value
