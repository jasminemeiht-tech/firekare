from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def build_group_folds(
    df: pd.DataFrame,
    n_splits: int,
    group_column: str,
    label_column: str = "injury_label",
    method: str = "group_kfold",
    random_seed: int = 42,
) -> pd.DataFrame:
    method = method.lower()
    if method == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    elif method == "group_kfold":
        splitter = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"unsupported split method: {method}")

    out = df.copy()
    out["fold"] = -1
    groups = out[group_column]
    labels = out[label_column]

    for fold, (_, val_idx) in enumerate(splitter.split(out, labels, groups=groups)):
        out.loc[out.index[val_idx], "fold"] = fold

    return out


def write_split_summary(df: pd.DataFrame, out_path: Path) -> None:
    fold_label_counts = {}
    for fold, fold_df in df.groupby("fold", dropna=False):
        fold_key = str(int(fold)) if pd.notna(fold) else "nan"
        fold_label_counts[fold_key] = fold_df["injury_label"].value_counts(dropna=False).to_dict()

    subject_df = (
        df.groupby("subject_id", as_index=False)
        .agg({"injury_label": "first", "fold": "first"})
        .reset_index(drop=True)
    )
    subject_fold_label_counts = {}
    for fold, fold_df in subject_df.groupby("fold", dropna=False):
        fold_key = str(int(fold)) if pd.notna(fold) else "nan"
        subject_fold_label_counts[fold_key] = fold_df["injury_label"].value_counts(dropna=False).to_dict()

    summary = {
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "label_counts": df["injury_label"].value_counts(dropna=False).to_dict(),
        "action_counts": df["action"].value_counts(dropna=False).to_dict(),
        "fold_counts": df["fold"].value_counts(dropna=False).sort_index().to_dict(),
        "fold_label_counts": fold_label_counts,
        "subject_fold_label_counts": subject_fold_label_counts,
    }
    if "landing_status" in df.columns:
        summary["landing_status_counts"] = df["landing_status"].value_counts(dropna=False).to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
