from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch


def set_seed(
    seed: int,
    *,
    deterministic: bool = True,
    deterministic_algorithms: bool = False,
    deterministic_warn_only: bool = True,
    cudnn_benchmark: bool = False,
    torch_num_threads: int | None = None,
) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic or deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSRARF_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark) if not deterministic else False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            bool(deterministic_algorithms),
            warn_only=bool(deterministic_warn_only),
        )
    if torch_num_threads is not None:
        torch.set_num_threads(max(int(torch_num_threads), 1))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

