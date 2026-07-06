from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .data_io import detect_landing_time, load_opensim_csv


@dataclass
class ManifestPaths:
    project_root: Path
    raw_root: Path
    outputs_root: Path
    metadata_csv: Path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_manifest_paths(config: dict) -> ManifestPaths:
    return ManifestPaths(
        project_root=Path(config["project"]["root"]),
        raw_root=Path(config["data"]["raw_root"]),
        outputs_root=Path(config["data"]["outputs_root"]),
        metadata_csv=Path(config["data"]["metadata_csv"]),
    )


def normalize_raw_path(path_str: str, raw_root: Path, stale_prefixes: Iterable[str]) -> str:
    if pd.isna(path_str):
        return path_str
    path = str(path_str)
    for prefix in stale_prefixes:
        if path.startswith(prefix):
            suffix = path[len(prefix):].lstrip("/")
            return str(raw_root / suffix)
    return path


def build_trial_manifest(config: dict) -> pd.DataFrame:
    paths = get_manifest_paths(config)
    df = pd.read_csv(paths.metadata_csv)

    stale_prefixes = config["data"].get("stale_prefixes", [])
    file_cols = ["ik_file", "id_file", "grf_file", "trc_file", "emg_file"]
    for col in file_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda value: normalize_raw_path(value, paths.raw_root, stale_prefixes)
            )

    if config["data"].get("keep_actions"):
        df = df[df["action"].isin(config["data"]["keep_actions"])].copy()

    for col in file_cols:
        if col in df.columns:
            df[f"{col}_exists"] = df[col].apply(lambda x: Path(x).exists() if pd.notna(x) else False)

    df["all_required_exist"] = df[
        [f"{col}_exists" for col in ["ik_file", "id_file", "grf_file", "emg_file"]]
    ].all(axis=1)

    if config["data"].get("require_complete_files", True):
        df = df[df["all_required_exist"]].copy()

    grf_threshold = float(config["signals"].get("grf_threshold", 20.0))
    landing_times = []
    landing_statuses = []
    for _, row in df.iterrows():
        grf_file = row.get("grf_file")
        if pd.isna(grf_file):
            landing_times.append(pd.NA)
            landing_statuses.append("missing_grf")
            continue
        grf_df = load_opensim_csv(Path(str(grf_file)), file_type="grf")
        landing_time = detect_landing_time(grf_df, threshold=grf_threshold)
        if landing_time is None:
            landing_times.append(pd.NA)
            landing_statuses.append("no_landing")
        else:
            landing_times.append(float(landing_time))
            landing_statuses.append("ok")

    df["landing_time"] = landing_times
    df["landing_status"] = landing_statuses

    if config["data"].get("filter_no_landing", True):
        df = df[df["landing_status"] == "ok"].copy()

    df["subject_id"] = df["subject_id"].astype(str).str.zfill(2)
    df["injury_label"] = pd.to_numeric(df["injury_label"], errors="coerce")
    df["action_index"] = df["action"].map({"SSC": 0, "VDJ": 1})

    return df.reset_index(drop=True)
