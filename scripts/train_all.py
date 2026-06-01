from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config, resolve_runtime_config  # noqa: E402


DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08", "METR-LA", "PEMS-BAY")
PROFILES = ("nofft", "fft001")
SUMMARY_FIELDS = (
    "dataset",
    "profile",
    "status",
    "config",
    "run_id",
    "run_dir",
    "avg_mae",
    "avg_rmse",
    "avg_mape",
    "h1_mae",
    "h3_mae",
    "h6_mae",
    "h12_mae",
    "elapsed_sec",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RARF on multiple datasets and summarize test metrics.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--profile", choices=("nofft", "fft001", "both"), default="both")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-tag", default=None, help="Shared suffix for generated run ids.")
    parser.add_argument("--summary-dir", default="output/runs/_summaries")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining runs after a failure.")
    return parser.parse_args()


def selected_profiles(profile: str) -> tuple[str, ...]:
    return PROFILES if profile == "both" else (profile,)


def config_path(config_dir: str, dataset: str, profile: str) -> Path:
    suffix = "" if profile == "nofft" else "_fft001"
    return ROOT / config_dir / f"{dataset}{suffix}.json"


def run_id(dataset: str, profile: str, tag: str) -> str:
    return f"{dataset.lower()}_rarf_{profile}_{tag}"


def run_dir_for(config_path_: Path, run_id_: str) -> Path:
    config = resolve_runtime_config(load_config(str(config_path_)))
    dataset = str(config["data"]["dataset"])
    seed = int(config["train"].get("seed", 1))
    output_root = Path(config["output"].get("root", "output"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / "runs" / dataset / "RARF" / f"seed_{seed}" / run_id_


def train_command(config_path_: Path, device: str | None, epochs: int | None, run_id_: str) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path_.relative_to(ROOT)),
        "--run-id",
        run_id_,
    ]
    if device is not None:
        command.extend(["--device", device])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    return command


def read_test_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test metrics: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_horizon = {str(row["horizon"]): row for row in rows}
    avg = by_horizon.get("avg")
    if avg is None:
        raise ValueError(f"Metrics file has no avg row: {path}")
    return {
        "avg_mae": avg["mae"],
        "avg_rmse": avg["rmse"],
        "avg_mape": avg["mape"],
        "h1_mae": by_horizon.get("1", {}).get("mae", ""),
        "h3_mae": by_horizon.get("3", {}).get("mae", ""),
        "h6_mae": by_horizon.get("6", {}).get("mae", ""),
        "h12_mae": by_horizon.get("12", {}).get("mae", ""),
    }


def write_summary(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    print("\nsummary:")
    print("dataset profile status    avg_mae   avg_rmse  avg_mape  run_id")
    for row in rows:
        print(
            f"{str(row['dataset']):<7} "
            f"{str(row['profile']):<7} "
            f"{str(row['status']):<8} "
            f"{str(row.get('avg_mae', '')):>8} "
            f"{str(row.get('avg_rmse', '')):>9} "
            f"{str(row.get('avg_mape', '')):>9} "
            f"{row['run_id']}"
        )


def main() -> int:
    args = parse_args()
    tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    summary_path = ROOT / args.summary_dir / f"rarf_all_{args.profile}_{tag}.csv"
    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        for profile in selected_profiles(args.profile):
            config_path_ = config_path(args.config_dir, dataset, profile)
            run_id_ = run_id(dataset, profile, tag)
            run_dir = run_dir_for(config_path_, run_id_)
            command = train_command(config_path_, args.device, args.epochs, run_id_)
            print("\n" + " ".join(command))

            row: dict[str, object] = {
                "dataset": dataset,
                "profile": profile,
                "status": "dry_run" if args.dry_run else "running",
                "config": str(config_path_.relative_to(ROOT)),
                "run_id": run_id_,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "elapsed_sec": "",
                "error": "",
            }
            start = time.time()
            try:
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
                    row.update(read_test_metrics(run_dir / "test_metrics_by_horizon.csv"))
                    row["status"] = "ok"
                    row["elapsed_sec"] = f"{time.time() - start:.1f}"
            except Exception as exc:
                row["status"] = "failed"
                row["elapsed_sec"] = f"{time.time() - start:.1f}"
                row["error"] = str(exc)
                rows.append(row)
                write_summary(summary_path, rows)
                if not args.continue_on_error:
                    print_summary(rows)
                    print(f"\nsummary saved to: {summary_path.relative_to(ROOT)}")
                    raise
                continue

            rows.append(row)
            write_summary(summary_path, rows)

    print_summary(rows)
    print(f"\nsummary saved to: {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
