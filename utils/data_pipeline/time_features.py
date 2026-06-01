from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from utils.dataset_specs import DatasetSpec

from .common import FEATURE_CHANNELS, _canonical_dataset_dir


def _datetime64_to_python_date(day_value: np.datetime64) -> date:
    return datetime.strptime(str(day_value), "%Y-%m-%d").date()


def _nth_weekday(year: int, month: int, weekday: int, ordinal: int) -> date:
    cursor = date(year, month, 1)
    offset = (weekday - cursor.weekday()) % 7
    cursor = cursor + timedelta(days=offset)
    cursor = cursor + timedelta(weeks=ordinal - 1)
    return cursor


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    offset = (cursor.weekday() - weekday) % 7
    return cursor - timedelta(days=offset)


def _apply_observed_holiday_rule(raw_day: date) -> date:
    if raw_day.weekday() == 5:
        return raw_day - timedelta(days=1)
    if raw_day.weekday() == 6:
        return raw_day + timedelta(days=1)
    return raw_day


def build_us_federal_observed_holidays(year: int) -> Dict[date, str]:
    holidays = {
        _apply_observed_holiday_rule(date(year, 1, 1)): "new_year_observed",
        _nth_weekday(year, 1, 0, 3): "martin_luther_king_jr_day",
        _nth_weekday(year, 2, 0, 3): "washingtons_birthday",
        _last_weekday(year, 5, 0): "memorial_day",
        _apply_observed_holiday_rule(date(year, 7, 4)): "independence_day_observed",
        _nth_weekday(year, 9, 0, 1): "labor_day",
        _nth_weekday(year, 10, 0, 2): "columbus_day",
        _apply_observed_holiday_rule(date(year, 11, 11)): "veterans_day_observed",
        _nth_weekday(year, 11, 3, 4): "thanksgiving_day",
        _apply_observed_holiday_rule(date(year, 12, 25)): "christmas_day_observed",
    }
    if year >= 2021:
        holidays[_apply_observed_holiday_rule(date(year, 6, 19))] = "juneteenth_observed"
    return holidays


def _build_holiday_lookup(unique_days: np.ndarray) -> Dict[date, str]:
    years = sorted({_datetime64_to_python_date(day_value).year for day_value in unique_days})
    lookup: Dict[date, str] = {}
    for year in years:
        lookup.update(build_us_federal_observed_holidays(year))
    return lookup


def _build_temporal_features(
    timestamps: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, Dict]:
    day_floor = timestamps.astype("datetime64[D]")
    tod = ((timestamps - day_floor) / np.timedelta64(1, "s")).astype(np.float32) / np.float32(86400.0)
    dow = ((day_floor.astype("int64") + 3) % 7).astype(np.float32)
    weekend = (dow >= 5).astype(np.float32)

    unique_days = np.unique(day_floor)
    holiday_lookup = _build_holiday_lookup(unique_days)
    holiday = np.array(
        [1.0 if _datetime64_to_python_date(day_value) in holiday_lookup else 0.0 for day_value in day_floor],
        dtype=np.float32,
    )
    holiday_names_in_range = sorted(
        {
            holiday_lookup[_datetime64_to_python_date(day_value)]
            for day_value in unique_days
            if _datetime64_to_python_date(day_value) in holiday_lookup
        }
    )

    tiled = {}
    for key, values in {
        "time_in_day": tod,
        "day_of_week": dow,
        "is_weekend": weekend,
        "is_holiday": holiday,
    }.items():
        tiled[key] = np.broadcast_to(values[:, None, None], (len(values), num_nodes, 1)).astype(np.float32, copy=False)

    metadata = {
        "sampling_interval_minutes": 5,
        "tod_definition": "seconds since local midnight divided by 86400; float in [0, 1)",
        "dow_definition": "Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6",
        "is_weekend_definition": "1 if day_of_week is Saturday or Sunday, else 0",
        "holiday_mode": "us_federal_observed",
        "holiday_source": "deterministic in-repository US federal observed holiday calendar rules",
        "holiday_status": "rule_based_us_federal_observed",
        "holiday_names_in_range": holiday_names_in_range,
    }
    return tiled, metadata


def _require_tables_for_hdf():
    try:
        import tables  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading METR-LA/PEMS-BAY raw .h5 files requires `tables`. "
            "Install the project preprocessing dependencies first."
        ) from exc
    return tables


def _resolve_hdf_key(raw_path: Path, configured_key: str | None) -> str:
    if configured_key:
        return configured_key
    tables = _require_tables_for_hdf()
    with tables.open_file(raw_path, mode="r") as handle:
        keys = [node._v_name for node in handle.root._f_list_nodes() if getattr(node, "_c_classid", "") == "GROUP"]
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"Unable to infer a unique HDF key for {raw_path}. "
        f"Available keys: {keys}. Please configure raw_hdf_key explicitly."
    )


def _decode_hdf_axis0(values: np.ndarray) -> list:
    decoded = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _load_hdf_fixed_group(raw_path: Path, hdf_key: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    tables = _require_tables_for_hdf()
    group_path = f"/{hdf_key.lstrip('/')}"
    with tables.open_file(raw_path, mode="r") as handle:
        axis0 = handle.get_node(f"{group_path}/axis0").read()
        axis1 = handle.get_node(f"{group_path}/axis1").read()
        values = handle.get_node(f"{group_path}/block0_values").read()
    traffic = np.asarray(values, dtype=np.float32)[..., None]
    timestamps = np.asarray(axis1, dtype="datetime64[ns]")
    metadata = {
        "raw_hdf_key": hdf_key.lstrip("/"),
        "raw_hdf_group_path": group_path,
        "raw_sensor_ids_preview": _decode_hdf_axis0(axis0[: min(len(axis0), 5)]),
        "raw_sensor_id_count": int(axis0.shape[0]),
        "raw_axis1_dtype": str(np.asarray(axis1).dtype),
        "raw_storage_layout": "pandas_fixed_hdf_axis_arrays",
    }
    return traffic, timestamps, metadata


def load_raw_traffic(spec: DatasetSpec, raw_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    dataset_raw_dir = _canonical_dataset_dir(raw_root, spec.canonical_name)
    raw_path = dataset_raw_dir / spec.raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset file: {raw_path}")

    if spec.raw_kind == "hdf":
        hdf_key = _resolve_hdf_key(raw_path, spec.raw_hdf_key)
        traffic, timestamps, hdf_metadata = _load_hdf_fixed_group(raw_path, hdf_key)
        metadata = {
            "raw_path": str(raw_path),
            "raw_kind": spec.raw_kind,
            "raw_shape": [int(traffic.shape[0]), int(traffic.shape[1])],
            "timestamp_mode": "datetime_index",
            "date_start": str(timestamps[0]).replace("T", " "),
            "date_end": str(timestamps[-1]).replace("T", " "),
            "raw_feature_names": list(spec.raw_feature_names),
            "target_feature_name": spec.target_feature_name,
            **hdf_metadata,
        }
        return traffic, timestamps, metadata

    data = np.load(raw_path)["data"].astype(np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.shape[1] != spec.num_nodes:
        raise ValueError(
            f"{spec.canonical_name} raw node count {data.shape[1]} does not match expected {spec.num_nodes}."
        )
    if data.shape[2] <= spec.primary_channel_index:
        raise ValueError(
            f"{spec.canonical_name} raw feature count {data.shape[2]} does not contain channel {spec.primary_channel_index}."
        )
    traffic = data[..., spec.primary_channel_index : spec.primary_channel_index + 1]
    start = np.datetime64(spec.synthetic_start_date.replace(" ", "T"))
    timestamps = start + np.arange(data.shape[0], dtype=np.int64) * np.timedelta64(5, "m")
    end = timestamps[-1]
    metadata = {
        "raw_path": str(raw_path),
        "raw_kind": spec.raw_kind,
        "raw_shape": list(data.shape),
        "raw_feature_names": list(spec.raw_feature_names),
        "primary_channel_index": spec.primary_channel_index,
        "target_feature_name": spec.target_feature_name,
        "timestamp_mode": "synthetic_start_date_assumption",
        "synthetic_start_date": spec.synthetic_start_date,
        "date_start": str(start).replace("T", " "),
        "date_end": str(end).replace("T", " "),
        "timestamp_note": (
            "Raw NPZ traffic benchmarks do not ship explicit timestamps. "
            "A synthetic local 5-minute calendar is anchored at a common benchmark start date assumption."
        ),
    }
    return traffic, timestamps, metadata


def build_processed_series(
    spec: DatasetSpec,
    raw_root: Path,
) -> Tuple[np.ndarray, Dict]:
    traffic, timestamps, raw_metadata = load_raw_traffic(spec, raw_root)
    num_steps, num_nodes, _ = traffic.shape
    temporal_features, temporal_metadata = _build_temporal_features(timestamps, num_nodes)

    series = np.concatenate(
        [
            traffic.astype(np.float32, copy=False),
            temporal_features["time_in_day"],
            temporal_features["day_of_week"],
            temporal_features["is_weekend"],
            temporal_features["is_holiday"],
        ],
        axis=-1,
    )
    metadata = {
        "dataset": spec.canonical_name,
        "num_timesteps": num_steps,
        "num_nodes": num_nodes,
        "num_features": int(series.shape[-1]),
        "feature_channels": list(FEATURE_CHANNELS),
        "feature_channel_indices": {name: idx for idx, name in enumerate(FEATURE_CHANNELS)},
        "target_feature_name": spec.target_feature_name,
        "timezone_used": spec.timezone,
        "timestamp_mode": spec.timestamp_mode,
        "raw": raw_metadata,
        "temporal_feature_metadata": temporal_metadata,
    }
    return series, metadata
