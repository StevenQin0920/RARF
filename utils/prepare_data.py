from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_pipeline import prepare_datasets


DEFAULT_DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def _run_regime_anchor_field_builder(dataset: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "utils.regime_anchor_field",
        "--dataset",
        dataset,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {"dataset": dataset, "status": "regime_anchor_field_built", "stdout": completed.stdout}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare RARF datasets and graph assets.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "datasets" / "raw_data"))
    parser.add_argument("--processed-root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--artifact-mode", choices=["series", "split_npz", "both"], default="split_npz")
    parser.add_argument("--npz-export-chunk-size", type=int, default=64)
    parser.add_argument(
        "--rarf-assets",
        action="store_true",
        help="Also build RARF regime anchor fields.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepared = prepare_datasets(
        dataset_names=args.datasets,
        raw_root=Path(args.raw_root),
        processed_root=Path(args.processed_root),
        split_root=Path(args.processed_root),
        history_len=args.history_length,
        horizon=args.horizon,
        artifact_mode=args.artifact_mode,
        npz_export_chunk_size=args.npz_export_chunk_size,
        force=args.force,
    )
    rarf_assets = []
    if args.rarf_assets:
        for dataset in args.datasets:
            rarf_assets.append(_run_regime_anchor_field_builder(dataset))

    print(json.dumps({"prepared": prepared, "rarf_assets": rarf_assets}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
