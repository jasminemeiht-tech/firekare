from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_emg_metadata(path: str | Path) -> dict[str, object]:
    meta: dict[str, object] = {}
    with Path(path).open("r", encoding="gbk", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 3:
                break
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                if key == "采样频率":
                    try:
                        meta["fs"] = float(value)
                    except ValueError:
                        pass
                elif key:
                    meta[key] = value
    return meta


def read_emg(path: str | Path) -> pd.DataFrame:
    """Read GBK CSV exported from xlsx/slk EMG files."""
    df = pd.read_csv(path, encoding="gbk", skiprows=3, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    first = str(df.columns[0])
    if first.lower().startswith("time") or "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if "time" not in df.columns:
        raise ValueError(f"EMG file has no time column: {path}")
    return df.sort_values("time").reset_index(drop=True)
