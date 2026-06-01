from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import build_regime_anchor_field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only daily and weekly-residual regime anchor fields.")
    parser.add_argument("--dataset", default="PEMS08")
    parser.add_argument("--train-npz", default=None)
    parser.add_argument("--anchor-asset-dir", default=None)
    parser.add_argument("--tod-size", type=int, default=288)
    parser.add_argument("--dow-size", type=int, default=7)
    args = parser.parse_args()

    dataset = args.dataset
    train_npz = Path(args.train_npz) if args.train_npz else REPO_ROOT / "datasets" / dataset / "train.npz"
    anchor_asset_dir = (
        Path(args.anchor_asset_dir)
        if args.anchor_asset_dir
        else REPO_ROOT / "datasets" / dataset / "anchors"
    )
    result = build_regime_anchor_field(
        dataset=dataset,
        train_npz=train_npz,
        anchor_asset_dir=anchor_asset_dir,
        tod_size=int(args.tod_size),
        dow_size=int(args.dow_size),
    )
    print(f"saved daily regime anchor field: {result['daily_path']} {result['daily_profile'].shape}")
    print(f"saved weekly residual regime anchor field: {result['weekly_path']} {result['weekly_profile'].shape}")
    print(f"saved metadata: {result['metadata_path']}")
