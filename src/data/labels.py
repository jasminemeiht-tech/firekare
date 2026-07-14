from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels(path: str | Path) -> pd.DataFrame:
    """Parse 标签.xlsx into [subj, label, VDJ_start, VDJ_end, SSC_start, SSC_end]."""
    raw = pd.read_excel(path, header=None)
    data = raw.iloc[2:, :6].copy()
    data.columns = ["label", "seq", "VDJ_start", "VDJ_end", "SSC_start", "SSC_end"]
    data = data.dropna(subset=["seq"])
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["label", "seq"])
    data["subj"] = data["seq"].astype(int).map(lambda x: f"{x:02d}")
    data["label"] = data["label"].astype(int)
    return (
        data[["subj", "label", "VDJ_start", "VDJ_end", "SSC_start", "SSC_end"]]
        .sort_values("subj")
        .reset_index(drop=True)
    )
