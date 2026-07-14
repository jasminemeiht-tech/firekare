from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pandas as pd


def _split(line: str) -> list[str]:
    return next(csv.reader([line.rstrip("\n")]))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    rename: dict[str, str] = {}
    for col in df.columns:
        name = str(col).strip()
        if name.lower() == "time":
            rename[col] = "time"
        elif name == "Frame#":
            rename[col] = "frame"
    df = df.rename(columns=rename)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if "time" not in df.columns:
        raise ValueError("motion file has no time column")
    return df.sort_values("time").reset_index(drop=True)


def _read_regular(lines: list[str], header_idx: int) -> pd.DataFrame:
    return _clean(pd.read_csv(StringIO("".join(lines[header_idx:]))))


def _read_trc(lines: list[str], header_idx: int) -> pd.DataFrame:
    marker = _split(lines[header_idx])
    axes = _split(lines[header_idx + 1]) if header_idx + 1 < len(lines) else []
    data_start = header_idx + 2

    current = ""
    cols: list[str] = []
    width = max(len(marker), len(axes))
    for i in range(width):
        m = marker[i].strip() if i < len(marker) else ""
        a = axes[i].strip() if i < len(axes) else ""
        if i == 0:
            cols.append("frame")
        elif i == 1:
            cols.append("time")
        else:
            if m:
                current = m
            cols.append(f"{current}_{a}" if current and a else f"col{i}")

    df = pd.read_csv(StringIO("".join(lines[data_start:])), header=None)
    df = df.iloc[:, : len(cols)]
    df.columns = cols[: df.shape[1]]
    return _clean(df)


def read_motion(path: str | Path) -> pd.DataFrame:
    """Read IK/ID/mot/TRC CSV exports with multi-line headers."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines(True)
    for i, line in enumerate(lines):
        parts = _split(line)
        if not parts:
            continue
        first = parts[0].strip().strip('"').lower()
        second = parts[1].strip().strip('"').lower() if len(parts) > 1 else ""
        if first == "time":
            return _read_regular(lines, i)
        if first == "frame#" and second == "time":
            return _read_trc(lines, i)
    raise ValueError(f"could not find motion header in {path}")
