from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.frozen_regime_anchor import FrozenRegimeAnchor  # noqa: E402
from utils.config import load_config, resolve_path, resolve_runtime_config, resolve_target_policy  # noqa: E402
from utils.experiment import load_data_resources, resolve_device, resolve_run_id  # noqa: E402
from utils.io import iter_batches  # noqa: E402
from utils.losses import compute_metrics  # noqa: E402
from utils.reporting import print_test_metrics_by_horizon, write_rows_csv  # noqa: E402
from utils.runtime import ensure_dir, set_seed  # noqa: E402


ANCHOR_ONLY_VARIANT = "anchor-only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the non-parametric Anchor Only baseline: "
            "Y_hat = A0_future with no learned residual correction."
        )
    )
    parser.add_argument("--config", required=True, help="RARF slim config path.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--run-id", default=None, help="Run id for the output directory.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit test batches for smoke checks.")
    parser.add_argument(
        "--include-padding",
        action="store_true",
        help="Keep dataloader padding samples. By default, metrics are trimmed to the true test split size.",
    )
    return parser.parse_args()


def build_anchor_only_run_dir(config: Dict[str, Any], run_id: str) -> Path:
    dataset = str(config["data"]["dataset"])
    seed = int(config["train"].get("seed", 1))
    output_root = Path(config["output"].get("root", "output"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / "runs" / dataset / "RARF_ABLATION" / ANCHOR_ONLY_VARIANT / f"seed_{seed}" / run_id


def build_anchor_only_model(config: Dict[str, Any], resources, device: torch.device) -> FrozenRegimeAnchor:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return FrozenRegimeAnchor(
        num_nodes=int(data_cfg["num_nodes"]),
        tod_vocab_size=int(model_cfg.get("tod_vocab_size", 288)),
        dow_vocab_size=int(model_cfg.get("dow_vocab_size", 7)),
        horizon=int(model_cfg.get("horizon", model_cfg.get("seq_length", 12))),
        daily_init_path=str(resolve_path(data_cfg["regime_anchor_field_daily_path"])),
        weekly_init_path=str(resolve_path(data_cfg["regime_anchor_field_weekly_path"])),
        daily_weight=float(model_cfg.get("regime_daily_weight", 1.0)),
        weekly_weight=float(model_cfg.get("regime_weekly_weight", 1.0)),
        spatial_bias_graph_adj=resources.prior_adj,
    ).to(device)


def future_temporal_ids(config: Dict[str, Any], future_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model_cfg = config["model"]
    tod_vocab_size = int(model_cfg.get("tod_vocab_size", 288))
    dow_vocab_size = int(model_cfg.get("dow_vocab_size", 7))
    tod_idx = int(model_cfg.get("future_context_tod_channel_index", 1))
    dow_idx = int(model_cfg.get("future_context_dow_channel_index", 2))
    required_idx = max(tod_idx, dow_idx)
    if future_data.shape[-1] <= required_idx:
        raise ValueError("future_data lacks required TOD/DOW channels for anchor lookup.")
    future_tod = torch.floor(future_data[:, :, 0, tod_idx] * tod_vocab_size).long()
    future_tod = future_tod.clamp(0, tod_vocab_size - 1)
    future_dow = future_data[:, :, 0, dow_idx].long().clamp(0, dow_vocab_size - 1)
    return future_tod, future_dow


def anchor_only_prediction(anchor: FrozenRegimeAnchor, config: Dict[str, Any], future_data: torch.Tensor) -> torch.Tensor:
    future_tod, future_dow = future_temporal_ids(config, future_data)
    daily_a0, weekly_a0 = anchor.lookup_a0_components(future_tod, future_dow)
    return (daily_a0 + weekly_a0).transpose(1, 2)


def trim_padding_if_needed(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    loader,
    include_padding: bool,
    max_eval_batches: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if include_padding or max_eval_batches is not None:
        return preds, labels
    original_size = int(getattr(loader, "original_size", preds.shape[0]))
    keep = min(original_size, int(preds.shape[0]))
    return preds[:keep], labels[:keep]


def evaluate_anchor_only(
    *,
    anchor: FrozenRegimeAnchor,
    config: Dict[str, Any],
    resources,
    device: torch.device,
    target_policy: Dict[str, Any],
    max_eval_batches: Optional[int],
    include_padding: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    anchor.eval()
    preds_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.inference_mode():
        for _, _, y in iter_batches(resources.test_loader, device, max_eval_batches):
            labels = y[:, :, :, 0].transpose(1, 2)
            preds = anchor_only_prediction(anchor, config, y)
            preds_list.append(resources.scaler.inverse_transform(preds).detach().cpu())
            labels_list.append(resources.scaler.inverse_transform(labels).detach().cpu())

    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    preds_all, labels_all = trim_padding_if_needed(
        preds_all,
        labels_all,
        loader=resources.test_loader,
        include_padding=include_padding,
        max_eval_batches=max_eval_batches,
    )

    rows: list[Dict[str, Any]] = []
    for horizon_idx in range(preds_all.shape[-1]):
        mae, mape, rmse = compute_metrics(
            preds_all[:, :, horizon_idx],
            labels_all[:, :, horizon_idx],
            mae_mask_value=target_policy["mae_mask_value"],
            rmse_mask_value=target_policy["rmse_mask_value"],
            mape_mask_value=target_policy["mape_mask_value"],
            mape_eps=float(target_policy["mape_eps"]),
        )
        rows.append({"horizon": horizon_idx + 1, "mae": mae, "rmse": rmse, "mape": mape})

    mae, mape, rmse = compute_metrics(
        preds_all,
        labels_all,
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    rows.append({"horizon": "avg", "mae": mae, "rmse": rmse, "mape": mape})

    summary = {
        "num_samples": int(labels_all.shape[0]),
        "num_nodes": int(labels_all.shape[1]),
        "horizon": int(labels_all.shape[2]),
        "include_padding": bool(include_padding),
        "max_eval_batches": max_eval_batches,
        "forecast_formula": "Y_hat = A0_future",
        "anchor_only_status": "non_parametric_train_only_frozen_anchor_lookup",
    }
    return rows, summary


def write_summary_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    source_config = load_config(args.config)
    config = resolve_runtime_config(source_config)
    target_policy = resolve_target_policy(config)
    train_cfg = config["train"]
    seed = int(train_cfg.get("seed", 1))
    set_seed(
        seed,
        deterministic=bool(train_cfg.get("deterministic", True)),
        deterministic_algorithms=bool(train_cfg.get("deterministic_algorithms", False)),
        deterministic_warn_only=bool(train_cfg.get("deterministic_warn_only", True)),
        cudnn_benchmark=bool(train_cfg.get("cudnn_benchmark", False)),
        torch_num_threads=train_cfg.get("torch_num_threads"),
    )

    device = resolve_device(args.device or train_cfg.get("device", "auto"))
    run_id = resolve_run_id(args.run_id)
    run_dir = build_anchor_only_run_dir(config, run_id)
    ensure_dir(run_dir)

    resources = load_data_resources(config)
    anchor = build_anchor_only_model(config, resources, device)
    rows, summary = evaluate_anchor_only(
        anchor=anchor,
        config=config,
        resources=resources,
        device=device,
        target_policy=target_policy,
        max_eval_batches=args.max_eval_batches,
        include_padding=bool(args.include_padding),
    )

    test_metrics_path = run_dir / "test_metrics_by_horizon.csv"
    summary_path = run_dir / "anchor_only_summary.json"
    write_rows_csv(test_metrics_path, rows)
    payload = {
        "dataset": str(config["data"]["dataset"]),
        "variant": ANCHOR_ONLY_VARIANT,
        "run_dir": str(run_dir),
        "test_metrics_by_horizon": str(test_metrics_path),
        "avg": next(row for row in rows if row["horizon"] == "avg"),
        **summary,
    }
    write_summary_json(summary_path, payload)

    print(f"variant: {ANCHOR_ONLY_VARIANT}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print_test_metrics_by_horizon(rows, "non_parametric_anchor_only")
    print(f"saved test metrics by horizon: {test_metrics_path}")
    print(f"saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
