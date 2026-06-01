from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Build RARF regime anchor assets.")
    parser.add_argument("--dataset", default="PEMS08")
    parser.add_argument("--tod-size", type=int, default=288)
    parser.add_argument("--dow-size", type=int, default=7)
    return parser.parse_args()


def cli() -> int:
    args = parse_args()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "utils.regime_anchor_field",
            "--dataset",
            args.dataset,
            "--tod-size",
            str(args.tod_size),
            "--dow-size",
            str(args.dow_size),
        ],
        cwd=ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
