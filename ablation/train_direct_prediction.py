from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.rarf_ablation import NoAnchorCoordinateTemporalBiasEncoder  # noqa: E402
from engine import RARFTrainer  # noqa: E402
from models.anchor_conditioned_residual_correction import TemporalBiasCorrectionBranch  # noqa: E402
from models.types import TensorAux  # noqa: E402
from utils.config import build_model_kwargs, load_config, resolve_runtime_config, resolve_target_policy  # noqa: E402
from utils.experiment import load_data_resources, resolve_device, resolve_run_id  # noqa: E402
from utils.reporting import print_test_metrics_by_horizon  # noqa: E402
from utils.runtime import ensure_dir, set_seed  # noqa: E402


DIRECT_VARIANT = "direct-prediction"


class DirectRawValuePredictionModel(nn.Module):
    """Direct full-signal prediction baseline for the anchor-residual reparameterization.

    This model intentionally does not instantiate FrozenRegimeAnchor, does not
    look up A0_future, and does not add an anchor to the decoder output. It
    keeps the history encoder, spatiotemporal refinement, and future-time-aware
    horizon decoder so that the comparison isolates the prediction
    parameterization rather than weakening the neural backbone.
    """

    PUBLIC_TENSOR_AUX_KEYS = {
        "direct_full_signal_prediction",
        "temporal_bias_correction",
    }
    GRAPH_AUX_KEYS = {"prior_adj", "physical_adj", "residual_support_graph"}

    def __init__(self, **model_args: object) -> None:
        super().__init__()
        self.forecast_mode = "direct_full_signal_prediction"
        self.num_nodes = int(model_args["num_nodes"])
        self.num_feat = int(model_args.get("num_feat", 1))
        self.horizon = int(model_args.get("horizon", model_args.get("seq_length", 12)))
        self.tod_vocab_size = int(model_args.get("tod_vocab_size", 288))
        self.dow_vocab_size = int(model_args.get("dow_vocab_size", 7))
        self.temporal_bias_encoder = NoAnchorCoordinateTemporalBiasEncoder(**model_args)
        self.temporal_bias_branch = TemporalBiasCorrectionBranch(**model_args)
        self.temporal_bias_encoder.reset_paper_parameters()

    def _temporal_id_sequences(self, history_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_idx = self.num_feat
        day_idx = self.num_feat + 1
        batch_size, seq_len, _, _ = history_data.shape
        if history_data.shape[-1] > time_idx:
            time_ids = torch.floor(history_data[:, :, :, time_idx] * self.tod_vocab_size).long()
            time_ids = time_ids.clamp(0, self.tod_vocab_size - 1)
        else:
            time_ids = torch.zeros(batch_size, seq_len, self.num_nodes, dtype=torch.long, device=history_data.device)
        if history_data.shape[-1] > day_idx:
            day_ids = history_data[:, :, :, day_idx].long().clamp(0, self.dow_vocab_size - 1)
        else:
            day_ids = torch.zeros(batch_size, seq_len, self.num_nodes, dtype=torch.long, device=history_data.device)
        return time_ids, day_ids

    def _zero_reference_tensors(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_length, num_nodes, _ = history_data.shape
        horizon = self.horizon if future_data is None else int(future_data.shape[1])
        history_reference = history_data.new_zeros((batch_size, history_length, num_nodes))
        future_reference = history_data.new_zeros((batch_size, horizon, num_nodes))
        return history_reference, future_reference

    @staticmethod
    def _minimal_aux(aux: TensorAux, keep_tensor_keys: Optional[Tuple[str, ...]]) -> TensorAux:
        keep = set(keep_tensor_keys or ())
        keep.add("direct_full_signal_prediction")
        return {key: value for key, value in aux.items() if key in keep}

    @classmethod
    def _public_aux(cls, aux: TensorAux, *, include_graphs: bool) -> TensorAux:
        keep = set(cls.PUBLIC_TENSOR_AUX_KEYS)
        if include_graphs:
            keep.update(cls.GRAPH_AUX_KEYS)
        keep.update(
            {
                "forecast_mode",
                "frozen_anchor_status",
                "anchor_coordinate_input_status",
                "anchor_coordinate_input_mode",
                "anchor_input_mode",
                "future_context_conditioning_status",
                "structure_encoder_status",
                "horizon_forecast_decoder_status",
                "input_missing_mask_status",
                "input_missing_mask_policy",
                "target_zero_is_valid",
                "temporal_encoder_type",
            }
        )
        return {key: value for key, value in aux.items() if key in keep}

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        return_graphs: bool = False,
        return_aux: bool = False,
        aux_mode: str = "full",
        aux_keep_keys: Optional[Tuple[str, ...]] = None,
    ):
        aux_mode = str(aux_mode).lower()
        if aux_mode not in {"full", "train_minimal"}:
            raise ValueError(f"aux_mode must be 'full' or 'train_minimal', got {aux_mode!r}.")

        history_tod, history_dow = self._temporal_id_sequences(history_data)
        zero_history_reference, zero_future_reference = self._zero_reference_tensors(history_data, future_data)
        encoded = self.temporal_bias_encoder(
            history_data,
            history_tod=history_tod,
            history_dow=history_dow,
            reference_daily_history=zero_history_reference,
            reference_weekly_history=zero_history_reference,
            reference_daily_future=zero_future_reference,
            reference_weekly_future=zero_future_reference,
        )
        decoded = self.temporal_bias_branch(
            encoded["structure_sequence"],
            encoded["temporal_state"],
            history_data,
            future_data,
            anchor_residual=encoded["anchor_residual"],
        )
        prediction = decoded["temporal_bias_correction"]
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("Direct prediction decoder did not produce a tensor forecast.")

        if return_graphs or return_aux:
            raw_aux: TensorAux = {
                **encoded["aux"],
                **decoded["aux"],
                "direct_full_signal_prediction": prediction,
                "forecast_mode": self.forecast_mode,
                "frozen_anchor_status": "not_used_direct_prediction",
                "anchor_coordinate_input_status": "disabled_direct_prediction",
                "anchor_coordinate_input_mode": "raw_history_direct_prediction",
                "anchor_input_mode": "raw_history_direct_prediction",
            }
            aux = self._public_aux(raw_aux, include_graphs=return_graphs)
            if aux_mode == "train_minimal":
                aux = self._minimal_aux(aux, aux_keep_keys)
            return prediction, aux
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the direct raw-value prediction ablation baseline.")
    parser.add_argument("--config", required=True, help="RARF slim config path.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--run-id", default=None, help="Run id for the output directory.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit val/test batches.")
    return parser.parse_args()


def build_direct_run_dir(config: Dict[str, Any], run_id: str) -> Path:
    dataset = str(config["data"]["dataset"])
    seed = int(config["train"].get("seed", 1))
    output_root = Path(config["output"].get("root", "output"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / "runs" / dataset / "RARF_ABLATION" / DIRECT_VARIANT / f"seed_{seed}" / run_id


def build_direct_model(
    config: Dict[str, Any],
    target_policy: Dict[str, Any],
    resources,
    device: torch.device,
) -> DirectRawValuePredictionModel:
    data_cfg = config["data"]
    missing_zero_value = float(data_cfg.get("missing_zero_value", 0.0))
    if bool(config["model"].get("use_input_missing_mask", False)) and target_policy["input_missing_mask_policy"] == "zero_as_missing":
        missing_zero_value = float(resources.scaler.transform(np.asarray([0.0], dtype=np.float32))[0])
    model_kwargs = build_model_kwargs(
        config,
        prior_adj=resources.prior_adj,
        physical_adj=resources.physical_adj,
        target_zero_is_valid=bool(target_policy["target_zero_is_valid"]),
        input_missing_mask_policy=str(target_policy["input_missing_mask_policy"]),
        missing_zero_value=missing_zero_value,
    )
    return DirectRawValuePredictionModel(**model_kwargs).to(device)


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
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    run_id = resolve_run_id(args.run_id)
    run_dir = build_direct_run_dir(config, run_id)
    ensure_dir(run_dir)

    resources = load_data_resources(config)
    model = build_direct_model(config, target_policy, resources, device)
    trainer = RARFTrainer(
        model=model,
        train_loader=resources.train_loader,
        val_loader=resources.val_loader,
        test_loader=resources.test_loader,
        scaler=resources.scaler,
        device=device,
        train_config=config["train"],
        eval_config=config["eval"],
        run_dir=run_dir,
    )

    print(f"variant: {DIRECT_VARIANT}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"fft_loss_weight: {float(config['train'].get('fft_loss_weight', 0.0))}")
    trainer.fit(
        epochs=epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        train_mask_value=target_policy["train_mask_value"],
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    test_rows, best_load_info = trainer.test(
        max_eval_batches=args.max_eval_batches,
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    print(f"saved checkpoint: {trainer.best_path}")
    print_test_metrics_by_horizon(test_rows, best_load_info.get("test_weight_source", "unknown_weights"))
    print(f"saved test metrics by horizon: {trainer.test_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
