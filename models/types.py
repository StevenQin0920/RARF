from __future__ import annotations

from typing import Any, Dict, Union

import torch

TensorAuxValue = Union[str, bool, int, float, torch.Tensor, list[Any], dict[str, Any], None]
TensorAux = Dict[str, TensorAuxValue]
