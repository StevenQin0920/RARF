from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    canonical_name: str
    aliases: Tuple[str, ...]
    raw_kind: str
    raw_filename: str
    num_nodes: int
    primary_channel_index: int
    split_ratios: Tuple[float, float, float]
    graph_kind: str
    graph_filename: str
    raw_hdf_key: Optional[str] = None
    graph_id_filename: Optional[str] = None
    graph_symmetrize: bool = False
    timezone: str = "America/Los_Angeles"
    timestamp_mode: str = "synthetic_start_date"
    synthetic_start_date: Optional[str] = None
    target_feature_name: str = "traffic"
    raw_feature_names: Tuple[str, ...] = ("traffic",)


_DATASET_SPECS = (
    DatasetSpec(
        canonical_name="METR-LA",
        aliases=("METR-LA", "METRLA", "metr-la", "metrla"),
        raw_kind="hdf",
        raw_filename="metr-la.h5",
        raw_hdf_key="df",
        num_nodes=207,
        primary_channel_index=0,
        split_ratios=(0.7, 0.1, 0.2),
        graph_kind="dcrnn_sensor_graph",
        graph_filename="adj_mx.pkl",
        timestamp_mode="datetime_index",
        target_feature_name="speed",
        raw_feature_names=("speed",),
    ),
    DatasetSpec(
        canonical_name="PEMS-BAY",
        aliases=("PEMS-BAY", "PEMSBAY", "pems-bay", "pemsbay"),
        raw_kind="hdf",
        raw_filename="pems-bay.h5",
        raw_hdf_key="speed",
        num_nodes=325,
        primary_channel_index=0,
        split_ratios=(0.7, 0.1, 0.2),
        graph_kind="dcrnn_sensor_graph",
        graph_filename="adj_mx_bay.pkl",
        timestamp_mode="datetime_index",
        target_feature_name="speed",
        raw_feature_names=("speed",),
    ),
    DatasetSpec(
        canonical_name="PEMS03",
        aliases=("PEMS03", "PEMS-03", "PEMSD3", "PEMSD03", "pems03", "pems-03"),
        raw_kind="npz",
        raw_filename="PEMS03.npz",
        num_nodes=358,
        primary_channel_index=0,
        split_ratios=(0.6, 0.2, 0.2),
        graph_kind="distance_csv",
        graph_filename="PEMS03.csv",
        graph_id_filename="PEMS03.txt",
        graph_symmetrize=True,
        synthetic_start_date="2018-09-01 00:00:00",
        target_feature_name="flow",
        raw_feature_names=("flow",),
    ),
    DatasetSpec(
        canonical_name="PEMS04",
        aliases=("PEMS04", "PEMS-04", "PEMSD4", "PEMSD04", "pems04", "pems-04"),
        raw_kind="npz",
        raw_filename="PEMS04.npz",
        num_nodes=307,
        primary_channel_index=0,
        split_ratios=(0.6, 0.2, 0.2),
        graph_kind="distance_csv",
        graph_filename="PEMS04.csv",
        graph_symmetrize=True,
        synthetic_start_date="2018-01-01 00:00:00",
        target_feature_name="flow",
        raw_feature_names=("flow", "occupancy", "speed"),
    ),
    DatasetSpec(
        canonical_name="PEMS07",
        aliases=("PEMS07", "PEMS-07", "PEMSD7", "PEMSD07", "pems07", "pems-07"),
        raw_kind="npz",
        raw_filename="PEMS07.npz",
        num_nodes=883,
        primary_channel_index=0,
        split_ratios=(0.6, 0.2, 0.2),
        graph_kind="distance_csv",
        graph_filename="PEMS07.csv",
        graph_symmetrize=True,
        synthetic_start_date="2017-05-01 00:00:00",
        target_feature_name="flow",
        raw_feature_names=("flow",),
    ),
    DatasetSpec(
        canonical_name="PEMS08",
        aliases=("PEMS08", "PEMS-08", "PEMSD8", "PEMSD08", "pems08", "pems-08"),
        raw_kind="npz",
        raw_filename="PEMS08.npz",
        num_nodes=170,
        primary_channel_index=0,
        split_ratios=(0.6, 0.2, 0.2),
        graph_kind="distance_csv",
        graph_filename="PEMS08.csv",
        graph_symmetrize=True,
        synthetic_start_date="2016-07-01 00:00:00",
        target_feature_name="flow",
        raw_feature_names=("flow", "occupancy", "speed"),
    ),
)

_SPEC_BY_ALIAS: Dict[str, DatasetSpec] = {}
for _spec in _DATASET_SPECS:
    for _alias in _spec.aliases:
        _SPEC_BY_ALIAS[_alias.lower()] = _spec


def get_dataset_spec(name: str) -> DatasetSpec:
    normalized = name.strip().lower()
    if normalized not in _SPEC_BY_ALIAS:
        supported = ", ".join(spec.canonical_name for spec in _DATASET_SPECS)
        raise KeyError(f"Unsupported dataset `{name}`. Supported datasets: {supported}")
    return _SPEC_BY_ALIAS[normalized]


def canonicalize_dataset_name(name: str) -> str:
    return get_dataset_spec(name).canonical_name


def supported_datasets() -> Tuple[str, ...]:
    return tuple(spec.canonical_name for spec in _DATASET_SPECS)


def iter_dataset_specs(names: Iterable[str]):
    seen = set()
    for name in names:
        spec = get_dataset_spec(name)
        if spec.canonical_name in seen:
            continue
        seen.add(spec.canonical_name)
        yield spec

