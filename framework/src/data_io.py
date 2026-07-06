from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def detect_header_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if "endheader" in line.lower():
                    return i + 1
    except UnicodeDecodeError:
        with path.open("r", encoding="latin1") as handle:
            for i, line in enumerate(handle):
                if "endheader" in line.lower():
                    return i + 1
    return 6


def load_opensim_csv(path: Path, file_type: str = "generic") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if file_type == "trc":
        df = pd.read_csv(path, skiprows=3, header=0)
        df = df.iloc[1:].reset_index(drop=True)
        df = df.dropna(axis=1, how="all")
        if "Time" in df.columns:
            df["time"] = pd.to_numeric(df["Time"], errors="coerce")
            df = df.drop(columns=["Time"])
        return df
    skiprows = detect_header_lines(path)
    df = pd.read_csv(path, skiprows=skiprows)
    df = df.dropna(axis=1, how="all")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def load_emg_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for encoding in ("utf-8", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(
                path,
                skiprows=3,
                encoding=encoding,
                on_bad_lines="skip",
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue
    return None


def detect_landing_time(grf_df: Optional[pd.DataFrame], threshold: float = 20.0) -> Optional[float]:
    if grf_df is None or grf_df.empty or "time" not in grf_df.columns:
        return None
    vz_cols = [col for col in grf_df.columns if col.endswith("_vz")]
    if not vz_cols:
        return None
    vz_df = grf_df[vz_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total_vz = vz_df.sum(axis=1)
    above = total_vz > threshold
    if not above.any():
        return None
    idx = above.idxmax()
    landing_time = pd.to_numeric(grf_df.loc[idx, "time"], errors="coerce")
    if pd.isna(landing_time) or not np.isfinite(float(landing_time)):
        return None
    return float(landing_time)
