from __future__ import annotations

import csv
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from utils.dataset_specs import DatasetSpec

from .common import _canonical_dataset_dir, _ensure_dir, _json_dump, _utc_timestamp


def _load_pickle(path: Path):
    visible_deprecation = getattr(np, "VisibleDeprecationWarning", None)
    if visible_deprecation is None:
        visible_deprecation = getattr(getattr(np, "exceptions", object), "VisibleDeprecationWarning", None)
    with open(path, "rb") as fp:
        with warnings.catch_warnings():
            if visible_deprecation is not None:
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed as Python or NumPy boolean but got `align=0`.*",
                    category=visible_deprecation,
                )
            try:
                return pickle.load(fp)
            except UnicodeDecodeError:
                fp.seek(0)
                return pickle.load(fp, encoding="latin1")


def _adjacency_stats(adj: np.ndarray) -> Dict:
    nonzero = int(np.count_nonzero(adj))
    return {
        "adjacency_shape": list(adj.shape),
        "adjacency_min": float(np.min(adj)),
        "adjacency_max": float(np.max(adj)),
        "nonzero_entries": nonzero,
        "density": float(nonzero / adj.size),
        "self_loop_nonzero_count": int(np.count_nonzero(np.diag(adj))),
        "symmetric": bool(np.allclose(adj, adj.T)),
        "symmetry_error_l1_mean": float(np.mean(np.abs(adj - adj.T))),
    }


def _build_distance_csv_adjacency(path: Path, num_nodes: int, id_path: Path | None, symmetrize: bool) -> Tuple[np.ndarray, Dict]:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    id_map = None
    if id_path is not None and id_path.exists():
        with open(id_path, "r", encoding="utf-8") as fp:
            ids = [line.strip() for line in fp if line.strip()]
        id_map = {int(sensor_id): idx for idx, sensor_id in enumerate(ids)}

    row_count = 0
    edge_count = 0
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if len(row) != 3:
                continue
            try:
                src = int(row[0])
                dst = int(row[1])
                _ = float(row[2])
            except ValueError:
                continue
            row_count += 1
            if id_map is not None:
                if src not in id_map or dst not in id_map:
                    continue
                src = id_map[src]
                dst = id_map[dst]
            if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
                continue
            adjacency[src, dst] = 1.0
            edge_count += 1
            if symmetrize:
                adjacency[dst, src] = 1.0

    metadata = {
        "source": str(path),
        "id_filename": str(id_path) if id_path is not None else None,
        "graph_symmetrize": symmetrize,
        "parsed_edge_rows": row_count,
        "stored_edges": edge_count,
    }
    metadata.update(_adjacency_stats(adjacency))
    return adjacency, metadata


def build_graph_artifacts(
    spec: DatasetSpec,
    raw_root: Path,
    processed_root: Path,
) -> Dict:
    graph_dir = _canonical_dataset_dir(processed_root, spec.canonical_name) / "graphs"
    _ensure_dir(graph_dir)
    if spec.graph_kind == "dcrnn_sensor_graph":
        raw_graph_path = raw_root / "sensor_graph" / spec.canonical_name / spec.graph_filename
        graph_obj = _load_pickle(raw_graph_path)
        adjacency = np.asarray(graph_obj[2] if isinstance(graph_obj, (tuple, list)) and len(graph_obj) >= 3 else graph_obj)
        adjacency = adjacency.astype(np.float32, copy=False)
        metadata = {
            "dataset": spec.canonical_name,
            "source": "DCRNN raw sensor graph pickle",
            "raw_graph_path": str(raw_graph_path),
            "graph_kind": spec.graph_kind,
            "graph_symmetrize": False,
        }
        metadata.update(_adjacency_stats(adjacency))
    elif spec.graph_kind == "distance_csv":
        raw_dataset_dir = _canonical_dataset_dir(raw_root, spec.canonical_name)
        csv_path = raw_dataset_dir / spec.graph_filename
        id_path = raw_dataset_dir / spec.graph_id_filename if spec.graph_id_filename else None
        adjacency, metadata = _build_distance_csv_adjacency(csv_path, spec.num_nodes, id_path, spec.graph_symmetrize)
        metadata.update(
            {
                "dataset": spec.canonical_name,
                "source": "distance CSV converted to binary physical adjacency",
                "graph_kind": spec.graph_kind,
            }
        )
    else:
        raise ValueError(f"Unsupported graph_kind `{spec.graph_kind}` for {spec.canonical_name}.")

    a_phy_path = graph_dir / "A_phy.pkl"
    a0_path = graph_dir / "A_0.pkl"
    with open(a_phy_path, "wb") as fp:
        pickle.dump(adjacency.astype(np.float32), fp)
    with open(a0_path, "wb") as fp:
        pickle.dump(adjacency.astype(np.float32), fp)

    graph_metadata = {
        "dataset": spec.canonical_name,
        "created_at_utc": _utc_timestamp(),
        "A_phy_path": str(a_phy_path),
        "A_0_path": str(a0_path),
        "A_0_status": "physical_prior_graph",
        **metadata,
    }
    _json_dump(graph_dir / "graph_metadata.json", graph_metadata)
    return graph_metadata
