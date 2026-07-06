from __future__ import annotations

import argparse
import copy
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import MultiModalTrialDataset, SubjectPairedTensorDataset
from .losses import build_binary_classification_loss
from .manifest import load_config
from .models import BiomarkerMLP, LinearResidualMLP, MultiModalCNN, PairedHybridModel


BIOMARKER_FEATURES = [
    "VDJ_emg_VLO_peak_t_ms",
    "VDJ_emg_SEMITEND_min",
    "SSC_imu_roll_deg_peak_t_ms",
    "VDJ_ik_hip_adduction_r_range",
    "SSC_syn2_early_integral",
    "VDJ_id_knee_angle_r_moment_min",
    "SSC_ik_hip_rotation_r_range",
    "VDJ_emg_VMO_max",
    "SSC_emg_MED. GASTRO_min",
    "SSC_id_hip_adduction_r_moment_std",
    "VDJ_ik_hip_adduction_r_std",
    "VDJ_ik_hip_adduction_r_abs_peak",
    "VDJ_syn1_sparsity",
    "VDJ_emg_VMO_abs_peak",
    "SSC_ik_subtalar_angle_r_std",
    "VDJ_label_duration",
    "VDJ_grf_grf2_vz_peak_t_ms",
    "VDJ_id_knee_angle_r_moment_mean",
    "VDJ_imu_Accel_Earth_Y_mG_abs_peak",
    "VDJ_id_hip_rotation_r_moment_mean",
]


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        return str(int(float(text)))
    except ValueError:
        return text.lstrip("0") or "0"


def split_action_biomarker_names(feature_names: List[str]) -> Tuple[List[str], List[str]]:
    ssc_names = [name for name in feature_names if str(name).startswith("SSC_")]
    vdj_names = [name for name in feature_names if str(name).startswith("VDJ_")]
    return ssc_names, vdj_names


def load_biomarker_feature_table(config: dict) -> pd.DataFrame:
    zip_path = Path(config["data"]["biomarker_package_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"biomarker feature package not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith("domain_landing_features.csv")]
        if not matches:
            raise FileNotFoundError(f"domain_landing_features.csv not found inside {zip_path}")
        with zf.open(matches[0]) as handle:
            feature_df = pd.read_csv(handle)

    missing = [name for name in BIOMARKER_FEATURES if name not in feature_df.columns]
    if missing:
        raise ValueError(f"missing biomarker columns in feature package: {missing[:5]}")

    if "subj" not in feature_df.columns or "label" not in feature_df.columns:
        raise ValueError("biomarker feature table must contain subj and label columns")

    table = feature_df[["subj", "label", *BIOMARKER_FEATURES]].copy()
    table["subject_id"] = table["subj"].map(normalize_subject_id)
    if table["subject_id"].duplicated().any():
        dup = table.loc[table["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"duplicated subjects in biomarker feature table: {dup[:5]}")

    for name in BIOMARKER_FEATURES:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table["label"] = pd.to_numeric(table["label"], errors="coerce")
    return table[["subject_id", "label", *BIOMARKER_FEATURES]].copy()


def load_subject_biomarker_features(
    config: dict,
    subject_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    feature_df = load_biomarker_feature_table(config)
    aligned = subject_df[["subject_id", "injury_label"]].copy()
    aligned["subject_id"] = aligned["subject_id"].map(normalize_subject_id)
    aligned = aligned.merge(feature_df, on="subject_id", how="left", sort=False)

    all_missing = aligned[BIOMARKER_FEATURES].isna().all(axis=1)
    if bool(all_missing.any()):
        missing_subjects = aligned.loc[all_missing, "subject_id"].tolist()
        raise ValueError(f"biomarker features missing for subjects: {missing_subjects[:5]}")

    label_mismatch = aligned["label"].notna() & (
        pd.to_numeric(aligned["label"], errors="coerce").astype(int) != aligned["injury_label"].astype(int)
    )
    if bool(label_mismatch.any()):
        bad = aligned.loc[label_mismatch, "subject_id"].tolist()
        raise ValueError(f"biomarker labels disagree with subject labels for subjects: {bad[:5]}")

    ssc_names, vdj_names = split_action_biomarker_names(BIOMARKER_FEATURES)
    return aligned, list(BIOMARKER_FEATURES), ssc_names, vdj_names


def load_domain_feature_table(config: dict) -> Tuple[pd.DataFrame, List[str]]:
    zip_path = Path(config["data"]["biomarker_package_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"domain feature package not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith("domain_landing_features.csv")]
        if not matches:
            raise FileNotFoundError(f"domain_landing_features.csv not found inside {zip_path}")
        with zf.open(matches[0]) as handle:
            feature_df = pd.read_csv(handle)

    if "subj" not in feature_df.columns or "label" not in feature_df.columns:
        raise ValueError("domain feature table must contain subj and label columns")

    table = feature_df.copy()
    table["subject_id"] = table["subj"].map(normalize_subject_id)
    if table["subject_id"].duplicated().any():
        dup = table.loc[table["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"duplicated subjects in domain feature table: {dup[:5]}")

    drop_cols = {"subj", "subject_id", "label"}
    feature_cols = [c for c in table.columns if c not in drop_cols]
    for name in feature_cols:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table["label"] = pd.to_numeric(table["label"], errors="coerce")
    return table[["subject_id", "label", *feature_cols]].copy(), feature_cols


def load_subject_domain_features(
    config: dict,
    subject_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_df, feature_cols = load_domain_feature_table(config)
    aligned = subject_df[["subject_id", "injury_label"]].copy()
    aligned["subject_id"] = aligned["subject_id"].map(normalize_subject_id)
    aligned = aligned.merge(feature_df, on="subject_id", how="left", sort=False)

    all_missing = aligned[feature_cols].isna().all(axis=1)
    if bool(all_missing.any()):
        missing_subjects = aligned.loc[all_missing, "subject_id"].tolist()
        raise ValueError(f"domain features missing for subjects: {missing_subjects[:5]}")

    label_mismatch = aligned["label"].notna() & (
        pd.to_numeric(aligned["label"], errors="coerce").astype(int) != aligned["injury_label"].astype(int)
    )
    if bool(label_mismatch.any()):
        bad = aligned.loc[label_mismatch, "subject_id"].tolist()
        raise ValueError(f"domain feature labels disagree with subject labels for subjects: {bad[:5]}")
    return aligned, feature_cols


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(config: dict) -> pd.DataFrame:
    manifest_path = Path(config["data"]["outputs_root"]) / "manifests" / "trials_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}. Run `python -m src.build_manifest --config config.yaml` first."
        )
    return pd.read_csv(manifest_path)


def ensure_output_dirs(config: dict) -> Dict[str, Path]:
    root = Path(config["data"]["outputs_root"])
    paths = {
        "root": root,
        "reports": root / "reports",
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def choose_device(config: dict) -> torch.device:
    requested = str(config["training"].get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_float(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return float(value)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, prob))
        metrics["auprc"] = float(average_precision_score(y_true, prob))
    else:
        metrics["auroc"] = math.nan
        metrics["auprc"] = math.nan

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics["accuracy"] = float((pred == y_true).mean())
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan
    metrics["f1"] = float(f1_score(y_true, pred, zero_division=0))
    metrics["tp"] = float(tp)
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    return metrics


def classification_metrics_from_predictions(
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = np.asarray(pred).astype(int)

    metrics: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, prob))
        metrics["auprc"] = float(average_precision_score(y_true, prob))
    else:
        metrics["auroc"] = math.nan
        metrics["auprc"] = math.nan

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics["accuracy"] = float((pred == y_true).mean())
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan
    metrics["f1"] = float(f1_score(y_true, pred, zero_division=0))
    metrics["tp"] = float(tp)
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    return metrics


def youden_index(metrics: Dict[str, float]) -> float:
    sensitivity = float(metrics.get("sensitivity", math.nan))
    specificity = float(metrics.get("specificity", math.nan))
    if not np.isfinite(sensitivity) or not np.isfinite(specificity):
        return math.nan
    return sensitivity + specificity - 1.0


def threshold_candidates(prob: np.ndarray, config: dict) -> np.ndarray:
    tuning_cfg = config.get("threshold_tuning", {})
    min_threshold = float(tuning_cfg.get("min_threshold", 0.05))
    max_threshold = float(tuning_cfg.get("max_threshold", 0.95))
    num_thresholds = int(tuning_cfg.get("num_thresholds", 181))
    grid = np.linspace(min_threshold, max_threshold, num_thresholds)
    prob = np.asarray(prob, dtype=float)
    empirical = np.clip(prob[np.isfinite(prob)], min_threshold, max_threshold)
    if empirical.size == 0:
        return grid
    return np.unique(np.concatenate([grid, empirical]))


def tune_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    strategy: str,
    config: dict,
) -> Tuple[float, float, Dict[str, float]]:
    strategy = strategy.lower()
    candidates = threshold_candidates(prob, config)
    best_threshold = float(config["training"].get("decision_threshold", 0.5))
    best_score = -math.inf
    best_metrics = classification_metrics(y_true, prob, threshold=best_threshold)

    for threshold in candidates:
        metrics = classification_metrics(y_true, prob, threshold=float(threshold))
        if strategy == "f1":
            score = float(metrics["f1"])
        elif strategy == "youden":
            score = float(youden_index(metrics))
        elif strategy == "accuracy":
            score = float(metrics["accuracy"])
        elif strategy in {"balanced_accuracy", "balacc"}:
            score = float(0.5 * (float(metrics["sensitivity"]) + float(metrics["specificity"])))
        else:
            raise ValueError(f"unsupported threshold strategy: {strategy}")

        is_better = score > best_score
        if not is_better and np.isfinite(score) and np.isclose(score, best_score):
            if abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                is_better = True

        if is_better:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def tune_threshold_fast(
    y_true: np.ndarray,
    prob: np.ndarray,
    strategy: str,
    config: dict,
) -> Tuple[float, float, Dict[str, float]]:
    strategy = strategy.lower()
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    candidates = threshold_candidates(prob, config)
    pred = prob[None, :] >= candidates[:, None]
    positives = y_true[None, :] == 1
    negatives = ~positives

    tp = np.sum(pred & positives, axis=1).astype(np.float64)
    fp = np.sum(pred & negatives, axis=1).astype(np.float64)
    fn = np.sum((~pred) & positives, axis=1).astype(np.float64)
    tn = np.sum((~pred) & negatives, axis=1).astype(np.float64)

    sensitivity = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan), where=(tn + fp) > 0)
    if strategy == "f1":
        scores = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)
    elif strategy == "youden":
        scores = sensitivity + specificity - 1.0
    elif strategy == "accuracy":
        scores = (tp + tn) / np.maximum(tp + tn + fp + fn, 1.0)
    elif strategy in {"balanced_accuracy", "balacc"}:
        scores = 0.5 * (sensitivity + specificity)
    else:
        raise ValueError(f"unsupported threshold strategy: {strategy}")

    finite_scores = np.where(np.isfinite(scores), scores, -math.inf)
    best_score = float(np.max(finite_scores))
    tie_idx = np.where(np.isclose(finite_scores, best_score))[0]
    if tie_idx.size == 0:
        best_idx = int(np.argmax(finite_scores))
    else:
        best_idx = int(tie_idx[np.argmin(np.abs(candidates[tie_idx] - 0.5))])
    best_threshold = float(candidates[best_idx])
    return best_threshold, best_score, classification_metrics(y_true, prob, threshold=best_threshold)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_platt_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    c_value: float = 1.0,
    enabled: bool = True,
) -> Dict[str, float | str]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels).astype(int)
    finite_mask = np.isfinite(logits.reshape(-1)) & np.isfinite(labels)
    logits = logits[finite_mask]
    labels = labels[finite_mask]
    if (not enabled) or len(np.unique(labels)) < 2 or logits.shape[0] < 4:
        return {"method": "identity", "coef": 1.0, "intercept": 0.0}

    try:
        calibrator = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            class_weight=None,
            max_iter=2000,
        )
        calibrator.fit(logits, labels)
        coef = float(calibrator.coef_[0, 0])
        intercept = float(calibrator.intercept_[0])
    except Exception:
        return {"method": "identity", "coef": 1.0, "intercept": 0.0}

    if not np.isfinite(coef) or not np.isfinite(intercept) or coef <= 1e-6:
        return {"method": "identity", "coef": 1.0, "intercept": 0.0, "raw_coef": safe_float(coef)}
    return {"method": "platt", "coef": coef, "intercept": intercept}


def apply_platt_calibrator(logits: np.ndarray, calibrator: Dict[str, float | str]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    coef = float(calibrator.get("coef", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return sigmoid_numpy(coef * logits + intercept)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_fold_table(df: pd.DataFrame) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for col in df.columns:
        if col == "fold":
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            summary[col] = {
                "mean": safe_float(numeric.mean()),
                "std": safe_float(numeric.std(ddof=1)),
                "min": safe_float(numeric.min()),
                "max": safe_float(numeric.max()),
            }
    return summary


def load_aligned_tabular_features(config: dict, manifest: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_paths = config["data"]["tabular_sources"]
    core = pd.read_csv(feature_paths["core_features"])
    mech = pd.read_csv(feature_paths["mechanics_features"]).drop(columns=["subject_id"], errors="ignore")
    synergy = pd.read_csv(feature_paths["synergy_features"])

    feature_df = core.merge(mech, on=["trial_id", "injury_label", "action"], how="inner")
    feature_df = feature_df.merge(synergy, on=["trial_id", "injury_label"], how="inner")
    if feature_df["trial_id"].duplicated().any():
        duplicates = feature_df.loc[feature_df["trial_id"].duplicated(), "trial_id"].tolist()
        raise ValueError(f"duplicated tabular feature rows found for trial_ids: {duplicates[:5]}")

    aligned = manifest[["trial_id", "subject_id", "action", "injury_label", "fold"]].merge(
        feature_df,
        on=["trial_id", "injury_label", "action"],
        how="left",
        sort=False,
    )
    numeric_cols = aligned.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = {"injury_label", "fold", "subject_id"}
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("no numeric tabular features available for hybrid model")

    all_missing_mask = aligned[feature_cols].isna().all(axis=1)
    if bool(all_missing_mask.any()):
        missing_trials = aligned.loc[all_missing_mask, "trial_id"].tolist()
        raise ValueError(f"tabular feature alignment failed for trials: {missing_trials[:5]}")
    return aligned, feature_cols


def action_slot_key(action: str) -> str:
    key = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(action))
    return key.strip("_")


def get_action_order(config: dict) -> List[str]:
    actions = [str(x) for x in config.get("data", {}).get("keep_actions", ["SSC", "VDJ"])]
    if len(actions) != 2:
        raise ValueError("subject-level paired model currently expects exactly 2 action slots")
    return actions


class TensorFoldDataset(Dataset):
    def __init__(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        tabular: torch.Tensor,
        action: torch.Tensor,
        label: torch.Tensor,
    ):
        self.emg = emg
        self.mechanics = mechanics
        self.tabular = tabular
        self.action = action
        self.label = label

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "emg": self.emg[index],
            "mechanics": self.mechanics[index],
            "tabular": self.tabular[index],
            "action": self.action[index],
            "label": self.label[index],
        }


class SubjectFeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        label: torch.Tensor,
    ):
        self.features = features
        self.label = label

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "label": self.label[index],
        }


@dataclass
class CachedTensors:
    emg: torch.Tensor
    mechanics: torch.Tensor
    tabular: torch.Tensor
    action: torch.Tensor
    label: torch.Tensor


@dataclass
class CachedSubjectTensors:
    emg: torch.Tensor
    mechanics: torch.Tensor
    tabular: torch.Tensor
    biomarker: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor
    label: torch.Tensor


def materialize_dataset(dataset: MultiModalTrialDataset, tabular: torch.Tensor) -> CachedTensors:
    if len(dataset) != int(tabular.shape[0]):
        raise ValueError("tabular feature rows must align one-to-one with manifest trials")
    emg_tensors: List[torch.Tensor] = []
    mech_tensors: List[torch.Tensor] = []
    action_tensors: List[torch.Tensor] = []
    label_tensors: List[torch.Tensor] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        emg_tensors.append(sample["emg"])
        mech_tensors.append(sample["mechanics"])
        action_tensors.append(sample["action"])
        label_tensors.append(sample["label"])
    return CachedTensors(
        emg=torch.stack(emg_tensors, dim=0),
        mechanics=torch.stack(mech_tensors, dim=0),
        tabular=tabular.clone(),
        action=torch.stack(action_tensors, dim=0),
        label=torch.stack(label_tensors, dim=0),
    )


def build_subject_table(manifest: pd.DataFrame, action_order: List[str]) -> pd.DataFrame:
    indexed = manifest.reset_index(drop=True).reset_index().rename(columns={"index": "trial_index"})
    rows: List[Dict[str, object]] = []
    for subject_id, group in indexed.groupby("subject_id", sort=True):
        labels = pd.to_numeric(group["injury_label"], errors="coerce").dropna().unique()
        folds = pd.to_numeric(group["fold"], errors="coerce").dropna().unique()
        if len(labels) != 1:
            raise ValueError(f"subject {subject_id} has inconsistent labels: {labels.tolist()}")
        if len(folds) != 1:
            raise ValueError(f"subject {subject_id} has inconsistent folds: {folds.tolist()}")

        row: Dict[str, object] = {
            "subject_id": str(subject_id),
            "injury_label": int(labels[0]),
            "fold": int(folds[0]),
        }
        for action in action_order:
            slot = action_slot_key(action)
            action_rows = group[group["action"] == action]
            if len(action_rows) > 1:
                trial_ids = action_rows["trial_id"].astype(str).tolist()
                raise ValueError(f"subject {subject_id} has multiple trials for action {action}: {trial_ids}")
            if len(action_rows) == 1:
                one = action_rows.iloc[0]
                row[f"trial_index_{slot}"] = int(one["trial_index"])
                row[f"trial_id_{slot}"] = str(one["trial_id"])
                row[f"has_{slot}"] = 1
            else:
                row[f"trial_index_{slot}"] = -1
                row[f"trial_id_{slot}"] = ""
                row[f"has_{slot}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def materialize_subject_pairs(
    subject_df: pd.DataFrame,
    cached: CachedTensors,
    action_order: List[str],
    action_name_to_index: Dict[str, int],
    biomarker: torch.Tensor,
) -> CachedSubjectTensors:
    n_subjects = len(subject_df)
    if n_subjects != int(biomarker.shape[0]):
        raise ValueError("subject biomarker rows must align one-to-one with subject_df")
    n_slots = len(action_order)
    _, emg_channels, target_points = cached.emg.shape
    _, mech_channels, _ = cached.mechanics.shape
    tabular_dim = int(cached.tabular.shape[1])

    emg = torch.zeros((n_subjects, n_slots, emg_channels, target_points), dtype=torch.float32)
    mechanics = torch.zeros((n_subjects, n_slots, mech_channels, target_points), dtype=torch.float32)
    tabular = torch.zeros((n_subjects, n_slots, tabular_dim), dtype=torch.float32)
    action = torch.zeros((n_subjects, n_slots), dtype=torch.long)
    action_mask = torch.zeros((n_subjects, n_slots), dtype=torch.float32)
    label = torch.as_tensor(subject_df["injury_label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    for slot_idx, action_name in enumerate(action_order):
        if action_name not in action_name_to_index:
            raise ValueError(f"missing action_index mapping for action {action_name}")
        action[:, slot_idx] = int(action_name_to_index[action_name])

    for subject_idx, row in subject_df.iterrows():
        for slot_idx, action_name in enumerate(action_order):
            slot = action_slot_key(action_name)
            trial_index = int(row[f"trial_index_{slot}"])
            if trial_index < 0:
                continue
            emg[subject_idx, slot_idx] = cached.emg[trial_index]
            mechanics[subject_idx, slot_idx] = cached.mechanics[trial_index]
            tabular[subject_idx, slot_idx] = cached.tabular[trial_index]
            action[subject_idx, slot_idx] = cached.action[trial_index]
            action_mask[subject_idx, slot_idx] = 1.0

    return CachedSubjectTensors(
        emg=emg,
        mechanics=mechanics,
        tabular=tabular,
        biomarker=biomarker.clone(),
        action=action,
        action_mask=action_mask,
        label=label,
    )


def compute_channel_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def compute_masked_channel_stats(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = mask[:, :, None, None].to(dtype=x.dtype)
    denom = weights.sum().clamp_min(1.0) * x.shape[-1]
    mean = (x * weights).sum(dim=(0, 1, 3), keepdim=True) / denom
    var = (((x - mean) ** 2) * weights).sum(dim=(0, 1, 3), keepdim=True) / denom
    std = torch.sqrt(var.clamp_min(1e-12))
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_fold_tensors(
    cached: CachedTensors,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[TensorFoldDataset, TensorFoldDataset, Dict[str, torch.Tensor]]:
    train_emg = cached.emg[train_idx].clone()
    train_mech = cached.mechanics[train_idx].clone()
    train_tab = cached.tabular[train_idx].clone()
    val_emg = cached.emg[val_idx].clone()
    val_mech = cached.mechanics[val_idx].clone()
    val_tab = cached.tabular[val_idx].clone()

    emg_mean, emg_std = compute_channel_stats(train_emg)
    mech_mean, mech_std = compute_channel_stats(train_mech)

    train_emg = (train_emg - emg_mean) / emg_std
    val_emg = (val_emg - emg_mean) / emg_std
    train_mech = (train_mech - mech_mean) / mech_std
    val_mech = (val_mech - mech_mean) / mech_std

    if train_tab.shape[1] > 0:
        imputer = SimpleImputer(strategy="median")
        train_tab_np = imputer.fit_transform(train_tab.numpy())
        val_tab_np = imputer.transform(val_tab.numpy())
        train_tab_np = np.nan_to_num(train_tab_np, nan=0.0).astype(np.float32, copy=False)
        val_tab_np = np.nan_to_num(val_tab_np, nan=0.0).astype(np.float32, copy=False)

        scaler = StandardScaler()
        train_tab_np = scaler.fit_transform(train_tab_np).astype(np.float32, copy=False)
        val_tab_np = scaler.transform(val_tab_np).astype(np.float32, copy=False)

        train_tab = torch.from_numpy(train_tab_np)
        val_tab = torch.from_numpy(val_tab_np)
        tabular_impute = torch.tensor(
            np.nan_to_num(np.asarray(imputer.statistics_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        tabular_mean = torch.tensor(
            np.nan_to_num(np.asarray(scaler.mean_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        tabular_std_np = np.nan_to_num(np.asarray(scaler.scale_, dtype=np.float32), nan=1.0)
        tabular_std_np[tabular_std_np < 1e-6] = 1.0
        tabular_std = torch.tensor(tabular_std_np, dtype=torch.float32)
    else:
        tabular_impute = torch.empty(0, dtype=torch.float32)
        tabular_mean = torch.empty(0, dtype=torch.float32)
        tabular_std = torch.empty(0, dtype=torch.float32)

    train_ds = TensorFoldDataset(
        emg=train_emg,
        mechanics=train_mech,
        tabular=train_tab,
        action=cached.action[train_idx].clone(),
        label=cached.label[train_idx].clone(),
    )
    val_ds = TensorFoldDataset(
        emg=val_emg,
        mechanics=val_mech,
        tabular=val_tab,
        action=cached.action[val_idx].clone(),
        label=cached.label[val_idx].clone(),
    )
    norm_state = {
        "emg_mean": emg_mean,
        "emg_std": emg_std,
        "mechanics_mean": mech_mean,
        "mechanics_std": mech_std,
        "tabular_impute": tabular_impute,
        "tabular_mean": tabular_mean,
        "tabular_std": tabular_std,
    }
    return train_ds, val_ds, norm_state


def normalize_paired_fold_tensors(
    cached: CachedSubjectTensors,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[SubjectPairedTensorDataset, SubjectPairedTensorDataset, Dict[str, torch.Tensor]]:
    train_emg = cached.emg[train_idx].clone()
    train_mech = cached.mechanics[train_idx].clone()
    train_tab = cached.tabular[train_idx].clone()
    train_biomarker = cached.biomarker[train_idx].clone()
    train_action = cached.action[train_idx].clone()
    train_mask = cached.action_mask[train_idx].clone()
    train_label = cached.label[train_idx].clone()

    val_emg = cached.emg[val_idx].clone()
    val_mech = cached.mechanics[val_idx].clone()
    val_tab = cached.tabular[val_idx].clone()
    val_biomarker = cached.biomarker[val_idx].clone()
    val_action = cached.action[val_idx].clone()
    val_mask = cached.action_mask[val_idx].clone()
    val_label = cached.label[val_idx].clone()

    emg_mean, emg_std = compute_masked_channel_stats(train_emg, train_mask)
    mech_mean, mech_std = compute_masked_channel_stats(train_mech, train_mask)

    train_emg = ((train_emg - emg_mean) / emg_std) * train_mask[:, :, None, None]
    val_emg = ((val_emg - emg_mean) / emg_std) * val_mask[:, :, None, None]
    train_mech = ((train_mech - mech_mean) / mech_std) * train_mask[:, :, None, None]
    val_mech = ((val_mech - mech_mean) / mech_std) * val_mask[:, :, None, None]

    if train_tab.shape[-1] > 0:
        train_mask_flat = train_mask.numpy().reshape(-1).astype(bool)
        val_mask_flat = val_mask.numpy().reshape(-1).astype(bool)
        train_flat = train_tab.numpy().reshape(-1, train_tab.shape[-1])
        val_flat = val_tab.numpy().reshape(-1, val_tab.shape[-1])

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_valid = train_flat[train_mask_flat]
        if train_valid.shape[0] == 0:
            raise ValueError("no valid paired tabular rows available in training fold")

        train_valid_imp = imputer.fit_transform(train_valid)
        train_valid_imp = np.nan_to_num(train_valid_imp, nan=0.0).astype(np.float32, copy=False)
        train_valid_scaled = scaler.fit_transform(train_valid_imp).astype(np.float32, copy=False)

        train_flat_out = np.zeros_like(train_flat, dtype=np.float32)
        val_flat_out = np.zeros_like(val_flat, dtype=np.float32)
        train_flat_out[train_mask_flat] = train_valid_scaled

        if val_mask_flat.any():
            val_valid_imp = imputer.transform(val_flat[val_mask_flat])
            val_valid_imp = np.nan_to_num(val_valid_imp, nan=0.0).astype(np.float32, copy=False)
            val_flat_out[val_mask_flat] = scaler.transform(val_valid_imp).astype(np.float32, copy=False)

        train_tab = torch.from_numpy(train_flat_out.reshape(train_tab.shape))
        val_tab = torch.from_numpy(val_flat_out.reshape(val_tab.shape))

        tabular_impute = torch.tensor(
            np.nan_to_num(np.asarray(imputer.statistics_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        tabular_mean = torch.tensor(
            np.nan_to_num(np.asarray(scaler.mean_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        tabular_std_np = np.nan_to_num(np.asarray(scaler.scale_, dtype=np.float32), nan=1.0)
        tabular_std_np[tabular_std_np < 1e-6] = 1.0
        tabular_std = torch.tensor(tabular_std_np, dtype=torch.float32)
    else:
        tabular_impute = torch.empty(0, dtype=torch.float32)
        tabular_mean = torch.empty(0, dtype=torch.float32)
        tabular_std = torch.empty(0, dtype=torch.float32)

    if train_biomarker.shape[-1] > 0:
        biomarker_imputer = SimpleImputer(strategy="median")
        biomarker_scaler = StandardScaler()
        train_biomarker_np = biomarker_imputer.fit_transform(train_biomarker.numpy())
        val_biomarker_np = biomarker_imputer.transform(val_biomarker.numpy())
        train_biomarker_np = np.nan_to_num(train_biomarker_np, nan=0.0).astype(np.float32, copy=False)
        val_biomarker_np = np.nan_to_num(val_biomarker_np, nan=0.0).astype(np.float32, copy=False)
        train_biomarker_np = biomarker_scaler.fit_transform(train_biomarker_np).astype(np.float32, copy=False)
        val_biomarker_np = biomarker_scaler.transform(val_biomarker_np).astype(np.float32, copy=False)
        train_biomarker = torch.from_numpy(train_biomarker_np)
        val_biomarker = torch.from_numpy(val_biomarker_np)
        biomarker_impute = torch.tensor(
            np.nan_to_num(np.asarray(biomarker_imputer.statistics_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        biomarker_mean = torch.tensor(
            np.nan_to_num(np.asarray(biomarker_scaler.mean_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        biomarker_std_np = np.nan_to_num(np.asarray(biomarker_scaler.scale_, dtype=np.float32), nan=1.0)
        biomarker_std_np[biomarker_std_np < 1e-6] = 1.0
        biomarker_std = torch.tensor(biomarker_std_np, dtype=torch.float32)
    else:
        biomarker_impute = torch.empty(0, dtype=torch.float32)
        biomarker_mean = torch.empty(0, dtype=torch.float32)
        biomarker_std = torch.empty(0, dtype=torch.float32)

    train_ds = SubjectPairedTensorDataset(
        emg=train_emg,
        mechanics=train_mech,
        tabular=train_tab,
        biomarker=train_biomarker,
        action=train_action,
        action_mask=train_mask,
        label=train_label,
    )
    val_ds = SubjectPairedTensorDataset(
        emg=val_emg,
        mechanics=val_mech,
        tabular=val_tab,
        biomarker=val_biomarker,
        action=val_action,
        action_mask=val_mask,
        label=val_label,
    )
    norm_state = {
        "emg_mean": emg_mean,
        "emg_std": emg_std,
        "mechanics_mean": mech_mean,
        "mechanics_std": mech_std,
        "tabular_impute": tabular_impute,
        "tabular_mean": tabular_mean,
        "tabular_std": tabular_std,
        "biomarker_impute": biomarker_impute,
        "biomarker_mean": biomarker_mean,
        "biomarker_std": biomarker_std,
    }
    return train_ds, val_ds, norm_state


def normalize_subject_feature_fold_tensors(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[SubjectFeatureDataset, SubjectFeatureDataset, Dict[str, torch.Tensor]]:
    train_x = features[train_idx].clone()
    val_x = features[val_idx].clone()

    if train_x.shape[-1] > 0:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_np = imputer.fit_transform(train_x.numpy())
        val_np = imputer.transform(val_x.numpy())
        train_np = np.nan_to_num(train_np, nan=0.0).astype(np.float32, copy=False)
        val_np = np.nan_to_num(val_np, nan=0.0).astype(np.float32, copy=False)
        train_np = scaler.fit_transform(train_np).astype(np.float32, copy=False)
        val_np = scaler.transform(val_np).astype(np.float32, copy=False)
        train_x = torch.from_numpy(train_np)
        val_x = torch.from_numpy(val_np)
        feature_impute = torch.tensor(
            np.nan_to_num(np.asarray(imputer.statistics_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        feature_mean = torch.tensor(
            np.nan_to_num(np.asarray(scaler.mean_, dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        feature_std_np = np.nan_to_num(np.asarray(scaler.scale_, dtype=np.float32), nan=1.0)
        feature_std_np[feature_std_np < 1e-6] = 1.0
        feature_std = torch.tensor(feature_std_np, dtype=torch.float32)
    else:
        feature_impute = torch.empty(0, dtype=torch.float32)
        feature_mean = torch.empty(0, dtype=torch.float32)
        feature_std = torch.empty(0, dtype=torch.float32)

    train_ds = SubjectFeatureDataset(
        features=train_x,
        label=labels[train_idx].clone(),
    )
    val_ds = SubjectFeatureDataset(
        features=val_x,
        label=labels[val_idx].clone(),
    )
    norm_state = {
        "feature_impute": feature_impute,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    return train_ds, val_ds, norm_state


def build_sequence_model(config: dict, tabular_dim: int) -> MultiModalCNN:
    emg_channels = len(config["signals"]["emg_channels"])
    mech_channels = (
        len(config["signals"]["ik_channels"])
        + len(config["signals"]["id_channels"])
        + len(config["signals"]["grf_channels"])
    )
    seq_cfg = config["model"]["sequence"]
    return MultiModalCNN(
        emg_channels=emg_channels,
        mech_channels=mech_channels,
        tabular_dim=tabular_dim,
        emg_hidden=seq_cfg["emg_hidden"],
        mech_hidden=seq_cfg["mech_hidden"],
        tabular_hidden=int(seq_cfg.get("tabular_hidden", 32)),
        fusion_hidden=seq_cfg["fusion_hidden"],
        action_embedding_dim=seq_cfg["action_embedding_dim"],
        dropout=seq_cfg["dropout"],
    )


def build_biomarker_mlp_model(config: dict, input_dim: int) -> BiomarkerMLP:
    mlp_cfg = config["model"]["biomarker_mlp"]
    hidden_dims = mlp_cfg.get("hidden_dims", [64, 32])
    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]
    return BiomarkerMLP(
        input_dim=input_dim,
        hidden_dims=[int(x) for x in hidden_dims],
        dropout=float(mlp_cfg.get("dropout", 0.2)),
    )


def build_linear_residual_biomarker_model(config: dict, input_dim: int) -> LinearResidualMLP:
    select_cfg = config.get("biomarker_feature_select", {})
    model_cfg = select_cfg.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [32, 16])
    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]
    return LinearResidualMLP(
        input_dim=input_dim,
        hidden_dims=[int(x) for x in hidden_dims],
        dropout=float(model_cfg.get("dropout", 0.15)),
        residual_scale=float(model_cfg.get("residual_scale", 0.35)),
    )


def resolve_subject_paired_model_spec(
    config: dict,
    biomarker_dim: int,
    ssc_biomarker_dim: int,
    vdj_biomarker_dim: int,
) -> Dict[str, int | str | bool]:
    paired_cfg = config["model"].get("subject_paired", {})
    raw_variant = str(paired_cfg.get("variant", paired_cfg.get("name", "slotmax"))).strip().lower()
    variant_aliases = {
        "paired_hybrid": "biomarker_hybrid",
        "hybrid": "biomarker_hybrid",
        "biomarker_hybrid": "biomarker_hybrid",
        "paired_biomarker_aux": "biomarker_aux",
        "raw_biomarker_aux": "biomarker_aux",
        "biomarker_aux": "biomarker_aux",
        "paired_slotmax": "slotmax",
        "slot_max": "slotmax",
        "slotmax": "slotmax",
    }
    variant = variant_aliases.get(raw_variant, raw_variant)
    if variant not in {"slotmax", "biomarker_hybrid", "biomarker_aux"}:
        raise ValueError(f"unsupported subject_paired variant: {raw_variant}")

    if variant == "biomarker_hybrid":
        effective_biomarker_dim = int(biomarker_dim)
        effective_ssc_dim = int(ssc_biomarker_dim)
        effective_vdj_dim = int(vdj_biomarker_dim)
    elif variant == "biomarker_aux":
        effective_biomarker_dim = 0
        effective_ssc_dim = int(ssc_biomarker_dim)
        effective_vdj_dim = int(vdj_biomarker_dim)
    else:
        effective_biomarker_dim = 0
        effective_ssc_dim = 0
        effective_vdj_dim = 0

    report_tag = str(paired_cfg.get("report_tag", variant)).strip().lower().replace(" ", "_")
    report_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in report_tag).strip("_")
    if not report_tag:
        report_tag = variant

    return {
        "variant": variant,
        "report_tag": report_tag,
        "uses_biomarker_input": bool(effective_biomarker_dim > 0),
        "biomarker_dim": int(effective_biomarker_dim),
        "ssc_biomarker_dim": int(effective_ssc_dim),
        "vdj_biomarker_dim": int(effective_vdj_dim),
        "biomarker_aux_dim": int(effective_ssc_dim + effective_vdj_dim),
    }


def build_subject_paired_model(
    config: dict,
    tabular_dim: int,
    biomarker_dim: int,
    ssc_biomarker_dim: int,
    vdj_biomarker_dim: int,
    action_order: List[str],
) -> PairedHybridModel:
    emg_channels = len(config["signals"]["emg_channels"])
    mech_channels = (
        len(config["signals"]["ik_channels"])
        + len(config["signals"]["id_channels"])
        + len(config["signals"]["grf_channels"])
    )
    model_spec = resolve_subject_paired_model_spec(config, biomarker_dim, ssc_biomarker_dim, vdj_biomarker_dim)
    seq_cfg = config["model"]["sequence"]
    paired_cfg = config["model"].get("subject_paired", {})
    subject_hidden = int(paired_cfg.get("subject_hidden", seq_cfg["fusion_hidden"]))
    fusion_hidden = int(paired_cfg.get("fusion_hidden", max(subject_hidden, 128)))
    biomarker_hidden = int(paired_cfg.get("biomarker_hidden", max(subject_hidden // 2, 32)))
    aux_hidden = int(paired_cfg.get("biomarker_aux_hidden", max(subject_hidden // 2, 32)))
    if "SSC" not in action_order or "VDJ" not in action_order:
        raise ValueError("subject paired biomarker hybrid expects SSC and VDJ in action_order")
    return PairedHybridModel(
        emg_channels=emg_channels,
        mech_channels=mech_channels,
        tabular_dim=tabular_dim,
        emg_hidden=seq_cfg["emg_hidden"],
        mech_hidden=seq_cfg["mech_hidden"],
        tabular_hidden=int(seq_cfg.get("tabular_hidden", 32)),
        fusion_hidden=seq_cfg["fusion_hidden"],
        action_embedding_dim=seq_cfg["action_embedding_dim"],
        dropout=float(paired_cfg.get("dropout", seq_cfg["dropout"])),
        subject_hidden=subject_hidden,
        n_action_heads=len(action_order),
        biomarker_dim=int(model_spec["biomarker_dim"]),
        biomarker_hidden=biomarker_hidden,
        subject_fusion_hidden=fusion_hidden,
        aux_hidden=aux_hidden,
        ssc_biomarker_dim=int(model_spec["ssc_biomarker_dim"]),
        vdj_biomarker_dim=int(model_spec["vdj_biomarker_dim"]),
        ssc_slot_idx=action_order.index("SSC"),
        vdj_slot_idx=action_order.index("VDJ"),
    )


def build_weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    labels = np.asarray(labels).astype(int)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]
    sample_weights = class_weights[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=axis)
    return np.nan_to_num(med, nan=0.0).astype(np.float64, copy=False)


def impute_scale_numpy(
    x_train: np.ndarray,
    x_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_imp = imputer.fit_transform(x_train)
    x_score_imp = imputer.transform(x_score)
    x_train_imp = np.nan_to_num(x_train_imp, nan=0.0).astype(np.float64, copy=False)
    x_score_imp = np.nan_to_num(x_score_imp, nan=0.0).astype(np.float64, copy=False)
    x_train_scaled = scaler.fit_transform(x_train_imp).astype(np.float64, copy=False)
    x_score_scaled = scaler.transform(x_score_imp).astype(np.float64, copy=False)
    return x_train_scaled, x_score_scaled


def stable_rank_domain_features(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    feature_names: List[str],
    config: dict,
    seed: int,
) -> pd.DataFrame:
    select_cfg = config.get("biomarker_feature_select", {})
    max_missing_rate = float(select_cfg.get("max_missing_rate", 0.3))
    min_std = float(select_cfg.get("min_std", 1e-8))
    repeats = int(select_cfg.get("selection_repeats", 100))
    train_fraction = float(select_cfg.get("selection_train_fraction", 0.8))
    top_n_per_repeat = int(select_cfg.get("selection_top_n_per_repeat", 80))

    x_train = np.asarray(features[train_idx], dtype=np.float64)
    y_train = np.asarray(labels[train_idx]).astype(int)
    finite = np.isfinite(x_train)
    missing_rate = 1.0 - finite.mean(axis=0)
    with np.errstate(all="ignore"):
        std = np.nanstd(np.where(finite, x_train, np.nan), axis=0)
    std = np.nan_to_num(std, nan=0.0)
    candidate_indices = np.where((missing_rate <= max_missing_rate) & (std > min_std))[0]
    if candidate_indices.size == 0:
        raise RuntimeError("no candidate domain features survived prefiltering")

    counts = np.zeros(candidate_indices.size, dtype=np.float64)
    effect_sum = np.zeros(candidate_indices.size, dtype=np.float64)
    signed_auc_sum = np.zeros(candidate_indices.size, dtype=np.float64)
    direction_sum = np.zeros(candidate_indices.size, dtype=np.float64)

    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        train_size=train_fraction,
        random_state=seed,
    )
    for sub_rel, _ in splitter.split(np.zeros(len(train_idx)), y_train):
        y_sub = y_train[sub_rel]
        if len(np.unique(y_sub)) < 2:
            continue
        x_sub = x_train[sub_rel][:, candidate_indices]
        effects = np.zeros(candidate_indices.size, dtype=np.float64)
        aucs = np.full(candidate_indices.size, np.nan, dtype=np.float64)
        for pos in range(candidate_indices.size):
            values = x_sub[:, pos]
            finite_mask = np.isfinite(values)
            if int(finite_mask.sum()) < 4:
                continue
            clean_values = values.copy()
            med = float(np.nanmedian(clean_values[finite_mask]))
            clean_values[~finite_mask] = med
            if float(np.std(clean_values)) <= min_std:
                continue
            try:
                auc = float(roc_auc_score(y_sub, clean_values))
            except Exception:
                continue
            aucs[pos] = auc
            effects[pos] = abs(auc - 0.5)

        valid = np.isfinite(aucs) & (effects > 0)
        if not valid.any():
            continue
        valid_positions = np.where(valid)[0]
        order = valid_positions[np.argsort(effects[valid_positions])[::-1]]
        chosen = order[: min(top_n_per_repeat, order.size)]
        counts[chosen] += 1.0
        effect_sum[chosen] += effects[chosen]
        signed_auc_sum[chosen] += aucs[chosen]
        direction_sum[chosen] += np.where(aucs[chosen] >= 0.5, 1.0, -1.0)

    selected_once = counts > 0
    if not selected_once.any():
        raise RuntimeError("stable feature selector did not select any features")

    freq = counts / float(max(repeats, 1))
    mean_effect = np.zeros_like(effect_sum)
    mean_auc = np.full_like(effect_sum, np.nan)
    direction_consistency = np.zeros_like(effect_sum)
    mean_effect[selected_once] = effect_sum[selected_once] / counts[selected_once]
    mean_auc[selected_once] = signed_auc_sum[selected_once] / counts[selected_once]
    direction_consistency[selected_once] = np.abs(direction_sum[selected_once]) / counts[selected_once]
    stable_score = freq * mean_effect * direction_consistency

    rows = []
    for local_pos, feature_idx in enumerate(candidate_indices):
        if counts[local_pos] <= 0:
            continue
        rows.append(
            {
                "feature_index": int(feature_idx),
                "feature": feature_names[int(feature_idx)],
                "selection_count": int(counts[local_pos]),
                "selection_frequency": float(freq[local_pos]),
                "mean_abs_auc_minus_0.5": float(mean_effect[local_pos]),
                "mean_auc": float(mean_auc[local_pos]),
                "direction_consistency": float(direction_consistency[local_pos]),
                "stable_score": float(stable_score[local_pos]),
                "missing_rate": float(missing_rate[int(feature_idx)]),
                "std": float(std[int(feature_idx)]),
            }
        )
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        ["stable_score", "selection_frequency", "mean_abs_auc_minus_0.5"],
        ascending=False,
    ).reset_index(drop=True)
    ranked["stable_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def select_uncorrelated_top_k(
    features: np.ndarray,
    train_idx: np.ndarray,
    ranked: pd.DataFrame,
    k: int,
    corr_threshold: float,
    initial_indices: List[int] | None = None,
) -> List[int]:
    initial_indices = [] if initial_indices is None else [int(i) for i in initial_indices]
    candidate_indices = ranked["feature_index"].astype(int).to_numpy()
    candidate_indices = np.asarray([int(i) for i in candidate_indices if int(i) not in set(initial_indices)])
    if len(initial_indices) >= int(k):
        return initial_indices[: int(k)]

    all_indices = np.asarray([*initial_indices, *candidate_indices.tolist()], dtype=int)
    x_train = np.asarray(features[train_idx][:, all_indices], dtype=np.float64)
    med = finite_nanmedian(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, med[None, :])
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train = (x_train - x_train.mean(axis=0)) / std

    selected_positions: List[int] = list(range(len(initial_indices)))
    selected_indices: List[int] = list(initial_indices)
    for offset, feature_idx in enumerate(candidate_indices):
        pos = len(initial_indices) + offset
        if len(selected_indices) >= int(k):
            break
        if selected_positions:
            corr = np.asarray(
                [
                    np.corrcoef(x_train[:, pos], x_train[:, selected_pos])[0, 1]
                    for selected_pos in selected_positions
                ],
                dtype=np.float64,
            )
            corr = np.nan_to_num(np.abs(corr), nan=0.0)
            if bool((corr > corr_threshold).any()):
                continue
        selected_positions.append(pos)
        selected_indices.append(int(feature_idx))

    if len(selected_indices) < int(k):
        for offset, feature_idx in enumerate(candidate_indices):
            if len(selected_indices) >= int(k):
                break
            if int(feature_idx) not in selected_indices:
                pos = len(initial_indices) + offset
                selected_positions.append(pos)
                selected_indices.append(int(feature_idx))
    return selected_indices[: int(k)]


def evaluate_feature_subset_logreg_oof(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    selected_indices: List[int],
    config: dict,
    seed: int,
) -> Dict[str, float]:
    select_cfg = config.get("biomarker_feature_select", {})
    inner_splits_requested = int(select_cfg.get("k_selection_inner_splits", 4))
    logreg_c = float(select_cfg.get("k_selection_logreg_c", 0.03))
    metric = str(select_cfg.get("k_selection_metric", "f1")).lower()
    y_train = np.asarray(labels[train_idx]).astype(int)
    class_counts = np.bincount(y_train, minlength=2)
    inner_splits = int(min(inner_splits_requested, class_counts.min()))
    if inner_splits < 2:
        raise RuntimeError("too few minority samples for K-selection inner OOF")

    x_train_all = np.asarray(features[train_idx][:, selected_indices], dtype=np.float64)
    oof_prob = np.full(len(train_idx), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for inner_train_rel, inner_val_rel in splitter.split(np.zeros(len(train_idx)), y_train):
        x_inner_train, x_inner_val = impute_scale_numpy(
            x_train_all[inner_train_rel],
            x_train_all[inner_val_rel],
        )
        y_inner_train = y_train[inner_train_rel]
        clf = LogisticRegression(
            C=logreg_c,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
        clf.fit(x_inner_train, y_inner_train)
        oof_prob[inner_val_rel] = clf.predict_proba(x_inner_val)[:, 1]

    if not np.isfinite(oof_prob).all():
        raise RuntimeError("K-selection OOF probabilities contain non-finite values")
    tuned_threshold, tuned_score, tuned_metrics = tune_threshold(
        y_train,
        oof_prob,
        strategy=metric,
        config=config,
    )
    base_metrics = classification_metrics(y_train, oof_prob, threshold=0.5)
    return {
        "inner_splits": float(inner_splits),
        "k_selection_metric": metric,
        "k_selection_score": float(tuned_score),
        "k_selection_threshold": float(tuned_threshold),
        "oof_auroc": float(base_metrics["auroc"]),
        "oof_auprc": float(base_metrics["auprc"]),
        "oof_accuracy_at_0.5": float(base_metrics["accuracy"]),
        "oof_f1_at_0.5": float(base_metrics["f1"]),
        "oof_accuracy_tuned": float(tuned_metrics["accuracy"]),
        "oof_f1_tuned": float(tuned_metrics["f1"]),
        "oof_youden_tuned": float(youden_index(tuned_metrics)),
    }


def choose_stable_feature_subset(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    feature_names: List[str],
    config: dict,
    seed: int,
) -> Tuple[List[int], pd.DataFrame, pd.DataFrame]:
    select_cfg = config.get("biomarker_feature_select", {})
    top_k_values = [int(k) for k in select_cfg.get("top_k_values", [20, 30, 40])]
    corr_threshold = float(select_cfg.get("corr_threshold", 0.9))
    ranked = stable_rank_domain_features(features, labels, train_idx, feature_names, config, seed=seed)
    anchor_current = bool(select_cfg.get("anchor_current_biomarkers", False))
    anchor_indices: List[int] = []
    if anchor_current:
        feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
        anchor_indices = [int(feature_to_index[name]) for name in BIOMARKER_FEATURES if name in feature_to_index]
        if len(anchor_indices) != len(BIOMARKER_FEATURES):
            missing = [name for name in BIOMARKER_FEATURES if name not in feature_to_index]
            raise ValueError(f"anchor biomarker features missing from domain table: {missing[:5]}")
        ranked_for_extras = ranked[~ranked["feature_index"].astype(int).isin(anchor_indices)].copy()
    else:
        ranked_for_extras = ranked

    k_rows = []
    selected_by_k: Dict[int, List[int]] = {}
    for k in top_k_values:
        selected_indices = select_uncorrelated_top_k(
            features,
            train_idx,
            ranked_for_extras,
            k=k,
            corr_threshold=corr_threshold,
            initial_indices=anchor_indices if anchor_current else None,
        )
        selected_by_k[int(k)] = selected_indices
        metrics = evaluate_feature_subset_logreg_oof(
            features,
            labels,
            train_idx,
            selected_indices,
            config,
            seed=seed + int(k) * 17,
        )
        k_rows.append(
            {
                "k": int(k),
                "n_selected": int(len(selected_indices)),
                "anchor_current_biomarkers": int(anchor_current),
                "n_anchor_features": int(len(anchor_indices)),
                **metrics,
            }
        )
    k_df = pd.DataFrame(k_rows)
    metric = str(select_cfg.get("k_selection_metric", "f1")).lower()
    metric_col = "k_selection_score"
    k_df = k_df.sort_values(
        [metric_col, "oof_auprc", "oof_auroc", "k"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    chosen_k = int(k_df.iloc[0]["k"])
    k_df["chosen"] = k_df["k"].astype(int) == chosen_k

    selected_indices = selected_by_k[chosen_k]
    selected_rank = {feature_idx: rank + 1 for rank, feature_idx in enumerate(selected_indices)}
    anchor_set = set(anchor_indices)
    ranked["selected"] = ranked["feature_index"].map(lambda x: int(x) in selected_rank)
    ranked["selected_rank"] = ranked["feature_index"].map(lambda x: selected_rank.get(int(x), math.nan))
    ranked["anchor_feature"] = ranked["feature_index"].map(lambda x: int(x) in anchor_set)
    ranked["chosen_k"] = chosen_k
    return selected_indices, ranked, k_df


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "features" in batch:
        return model(batch["features"])
    kwargs = {
        "emg": batch["emg"],
        "mechanics": batch["mechanics"],
        "action": batch["action"],
        "tabular": batch["tabular"],
    }
    if "biomarker" in batch:
        kwargs["biomarker"] = batch["biomarker"]
    if "action_mask" in batch:
        kwargs["action_mask"] = batch["action_mask"]
    return model(**kwargs)


def forward_slot_logits_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not hasattr(model, "forward_slot_logits"):
        raise AttributeError("model does not expose forward_slot_logits")
    return model.forward_slot_logits(
        emg=batch["emg"],
        mechanics=batch["mechanics"],
        action=batch["action"],
        tabular=batch["tabular"],
    )


def forward_subject_details_from_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "forward_subject_details"):
        raise AttributeError("model does not expose forward_subject_details")
    return model.forward_subject_details(
        emg=batch["emg"],
        mechanics=batch["mechanics"],
        action=batch["action"],
        tabular=batch["tabular"],
        biomarker=batch.get("biomarker"),
        action_mask=batch.get("action_mask"),
    )


def build_subject_paired_aux_config(
    config: dict,
    action_order: List[str],
    biomarker_feature_names: List[str],
) -> Dict[str, float | int | bool | None]:
    aux_cfg = config.get("training", {}).get("subject_paired_aux", {})
    ssc_slot_idx = action_order.index("SSC") if "SSC" in action_order else None
    vdj_slot_idx = action_order.index("VDJ") if "VDJ" in action_order else None
    ssc_biomarker_names, vdj_biomarker_names = split_action_biomarker_names(biomarker_feature_names)
    ranking_weight = float(aux_cfg.get("ranking_weight", 0.0))
    hard_negative_weight = float(aux_cfg.get("hard_negative_weight", 0.0))
    ssc_weight = float(aux_cfg.get("ssc_negative_weight", 0.0))
    enabled = bool(aux_cfg.get("enabled", True)) and (
        ranking_weight > 0.0 or hard_negative_weight > 0.0 or ssc_weight > 0.0
    )
    return {
        "enabled": enabled,
        "ranking_weight": ranking_weight,
        "ranking_margin": float(aux_cfg.get("ranking_margin", 0.25)),
        "hard_negative_weight": hard_negative_weight,
        "hard_negative_margin": float(aux_cfg.get("hard_negative_margin", 0.0)),
        "hard_negative_topk": int(aux_cfg.get("hard_negative_topk", 2)),
        "hard_negative_warmup_epochs": int(aux_cfg.get("hard_negative_warmup_epochs", 0)),
        "ssc_negative_weight": ssc_weight,
        "ssc_negative_margin": float(aux_cfg.get("ssc_negative_margin", 0.0)),
        "ssc_negative_warmup_epochs": int(aux_cfg.get("ssc_negative_warmup_epochs", 0)),
        "ssc_slot_idx": ssc_slot_idx,
        "vdj_slot_idx": vdj_slot_idx,
        "raw_cls_weight": float(aux_cfg.get("raw_cls_weight", 0.5)),
        "biomarker_cls_weight": float(aux_cfg.get("biomarker_cls_weight", 0.25)),
        "biomarker_aux_weight": float(aux_cfg.get("biomarker_aux_weight", 0.2)),
        "biomarker_aux_delta": float(aux_cfg.get("biomarker_aux_delta", 1.0)),
        "biomarker_ssc_indices": [int(biomarker_feature_names.index(name)) for name in ssc_biomarker_names],
        "biomarker_vdj_indices": [int(biomarker_feature_names.index(name)) for name in vdj_biomarker_names],
        "biomarker_ssc_dim": len(ssc_biomarker_names),
        "biomarker_vdj_dim": len(vdj_biomarker_names),
    }


def resolve_subject_paired_aux_for_epoch(
    aux_cfg: Dict[str, float | int | bool | None],
    epoch: int,
) -> Dict[str, float | int | bool | None]:
    resolved = dict(aux_cfg)
    if int(epoch) < int(aux_cfg.get("hard_negative_warmup_epochs", 0)):
        resolved["hard_negative_weight"] = 0.0
    if int(epoch) < int(aux_cfg.get("ssc_negative_warmup_epochs", 0)):
        resolved["ssc_negative_weight"] = 0.0
    resolved["enabled"] = bool(aux_cfg.get("enabled", False)) and (
        float(resolved.get("ranking_weight", 0.0)) > 0.0
        or float(resolved.get("hard_negative_weight", 0.0)) > 0.0
        or float(resolved.get("ssc_negative_weight", 0.0)) > 0.0
    )
    return resolved


def _topk_mean(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    k = max(1, min(int(k), int(values.numel())))
    return torch.topk(values.reshape(-1), k=k, largest=True).values.mean()


def _huber_mean(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    delta = float(max(delta, 1e-6))
    error = torch.abs(pred - target)
    quadratic = torch.clamp(error, max=delta)
    linear = error - quadratic
    loss = 0.5 * quadratic.pow(2) / delta + linear
    return loss.mean()


def compute_subject_paired_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    aux_cfg: Dict[str, float | int | bool | None],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    details = forward_subject_details_from_batch(model, batch)
    subject_logits = details["subject_logit"]
    raw_subject_logits = details.get("raw_subject_logit", subject_logits)
    biomarker_logits = details.get("biomarker_logit")
    slot_logits = details["slot_logits"]
    labels = batch["label"].float()
    action_mask = batch.get("action_mask")
    if action_mask is None:
        action_mask = torch.ones_like(slot_logits, dtype=slot_logits.dtype, device=slot_logits.device)
    else:
        action_mask = action_mask.to(device=slot_logits.device, dtype=slot_logits.dtype)

    cls_loss = criterion(subject_logits, labels)
    total_loss = cls_loss
    raw_cls_loss = subject_logits.new_zeros(())
    biomarker_cls_loss = subject_logits.new_zeros(())
    biomarker_aux_loss = subject_logits.new_zeros(())
    biomarker_aux_ssc_loss = subject_logits.new_zeros(())
    biomarker_aux_vdj_loss = subject_logits.new_zeros(())

    rank_loss = subject_logits.new_zeros(())
    hard_neg_loss = subject_logits.new_zeros(())
    ssc_neg_loss = subject_logits.new_zeros(())

    raw_cls_weight = float(aux_cfg.get("raw_cls_weight", 0.0))
    if raw_cls_weight > 0.0:
        raw_cls_loss = criterion(raw_subject_logits, labels)
        total_loss = total_loss + raw_cls_weight * raw_cls_loss

    biomarker_cls_weight = float(aux_cfg.get("biomarker_cls_weight", 0.0))
    if biomarker_cls_weight > 0.0 and biomarker_logits is not None and "biomarker" in batch:
        biomarker_cls_loss = criterion(biomarker_logits, labels)
        total_loss = total_loss + biomarker_cls_weight * biomarker_cls_loss

    biomarker_aux_weight = float(aux_cfg.get("biomarker_aux_weight", 0.0))
    biomarker_target = batch.get("biomarker")
    if biomarker_aux_weight > 0.0 and biomarker_target is not None:
        biomarker_losses = []
        delta = float(aux_cfg.get("biomarker_aux_delta", 1.0))
        ssc_indices = aux_cfg.get("biomarker_ssc_indices", [])
        ssc_slot_idx = aux_cfg.get("ssc_slot_idx")
        pred_ssc = details.get("biomarker_pred_ssc")
        if pred_ssc is not None and pred_ssc.ndim == 2 and pred_ssc.shape[1] > 0 and ssc_slot_idx is not None and ssc_indices:
            ssc_valid = action_mask[:, int(ssc_slot_idx)] > 0
            if int(ssc_valid.sum().item()) > 0:
                target_ssc = biomarker_target[:, ssc_indices]
                biomarker_aux_ssc_loss = _huber_mean(pred_ssc[ssc_valid], target_ssc[ssc_valid], delta=delta)
                biomarker_losses.append(biomarker_aux_ssc_loss)

        vdj_indices = aux_cfg.get("biomarker_vdj_indices", [])
        vdj_slot_idx = aux_cfg.get("vdj_slot_idx")
        pred_vdj = details.get("biomarker_pred_vdj")
        if pred_vdj is not None and pred_vdj.ndim == 2 and pred_vdj.shape[1] > 0 and vdj_slot_idx is not None and vdj_indices:
            vdj_valid = action_mask[:, int(vdj_slot_idx)] > 0
            if int(vdj_valid.sum().item()) > 0:
                target_vdj = biomarker_target[:, vdj_indices]
                biomarker_aux_vdj_loss = _huber_mean(pred_vdj[vdj_valid], target_vdj[vdj_valid], delta=delta)
                biomarker_losses.append(biomarker_aux_vdj_loss)

        if biomarker_losses:
            biomarker_aux_loss = torch.stack(biomarker_losses).mean()
            total_loss = total_loss + biomarker_aux_weight * biomarker_aux_loss

    if bool(aux_cfg.get("enabled", False)):
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        has_pos = int(pos_mask.sum().item()) > 0
        has_neg = int(neg_mask.sum().item()) > 0

        ranking_weight = float(aux_cfg.get("ranking_weight", 0.0))
        if ranking_weight > 0.0 and has_pos and has_neg:
            pos_logits = raw_subject_logits[pos_mask][:, None]
            neg_logits = raw_subject_logits[neg_mask][None, :]
            margin = float(aux_cfg.get("ranking_margin", 0.25))
            rank_loss = torch.relu(margin - (pos_logits - neg_logits)).mean()
            total_loss = total_loss + ranking_weight * rank_loss

        hard_negative_weight = float(aux_cfg.get("hard_negative_weight", 0.0))
        if hard_negative_weight > 0.0 and has_neg:
            neg_subject_logits = raw_subject_logits[neg_mask]
            hard_margin = float(aux_cfg.get("hard_negative_margin", 0.0))
            hard_values = torch.relu(neg_subject_logits - hard_margin)
            hard_neg_loss = _topk_mean(hard_values, int(aux_cfg.get("hard_negative_topk", 2)))
            total_loss = total_loss + hard_negative_weight * hard_neg_loss

        ssc_weight = float(aux_cfg.get("ssc_negative_weight", 0.0))
        ssc_slot_idx = aux_cfg.get("ssc_slot_idx")
        if ssc_weight > 0.0 and ssc_slot_idx is not None and has_neg:
            ssc_valid = neg_mask & (action_mask[:, int(ssc_slot_idx)] > 0)
            if int(ssc_valid.sum().item()) > 0:
                ssc_logits = slot_logits[ssc_valid, int(ssc_slot_idx)]
                ssc_margin = float(aux_cfg.get("ssc_negative_margin", 0.0))
                ssc_values = torch.relu(ssc_logits - ssc_margin)
                ssc_neg_loss = _topk_mean(ssc_values, int(aux_cfg.get("hard_negative_topk", 2)))
                total_loss = total_loss + ssc_weight * ssc_neg_loss

    component_values = {
        "loss_cls": float(cls_loss.detach().item()),
        "loss_cls_raw": float(raw_cls_loss.detach().item()),
        "loss_cls_biomarker": float(biomarker_cls_loss.detach().item()),
        "loss_biomarker_aux": float(biomarker_aux_loss.detach().item()),
        "loss_biomarker_aux_ssc": float(biomarker_aux_ssc_loss.detach().item()),
        "loss_biomarker_aux_vdj": float(biomarker_aux_vdj_loss.detach().item()),
        "loss_rank": float(rank_loss.detach().item()),
        "loss_hard_neg": float(hard_neg_loss.detach().item()),
        "loss_ssc_neg": float(ssc_neg_loss.detach().item()),
    }
    return total_loss, subject_logits, component_values


def infer_sequence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = forward_from_batch(model, batch)
            if criterion is not None:
                loss = criterion(logits, batch["label"])
                losses.append(float(loss.item()))
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            label = batch["label"].detach().cpu().numpy()
            probs.append(prob)
            labels.append(label)
    mean_loss = float(np.mean(losses)) if losses else math.nan
    return mean_loss, np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)


def infer_sequence_with_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    probs: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = forward_from_batch(model, batch)
            if criterion is not None:
                loss = criterion(logits, batch["label"])
                losses.append(float(loss.item()))
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            label = batch["label"].detach().cpu().numpy()
            probs.append(prob)
            logits_list.append(logits.detach().cpu().numpy())
            labels.append(label)
    mean_loss = float(np.mean(losses)) if losses else math.nan
    return (
        mean_loss,
        np.concatenate(labels, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(logits_list, axis=0),
    )


def infer_paired_slot_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    if not hasattr(model, "forward_slot_logits"):
        raise AttributeError("paired slot probability inference requires a model with forward_slot_logits")
    model.eval()
    slot_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            slot_logits = forward_slot_logits_from_batch(model, batch)
            slot_prob = torch.sigmoid(slot_logits)
            if "action_mask" in batch:
                mask = batch["action_mask"].to(device=slot_prob.device, dtype=slot_prob.dtype)
                slot_prob = torch.where(mask > 0, slot_prob, torch.full_like(slot_prob, float("nan")))
            slot_probs.append(slot_prob.detach().cpu().numpy())
    return np.concatenate(slot_probs, axis=0)


def infer_subject_paired(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    aux_cfg: Dict[str, float | int | bool | None],
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    losses: List[float] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    component_rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            loss, logits, component_values = compute_subject_paired_loss(model, batch, criterion, aux_cfg)
            losses.append(float(loss.item()))
            component_rows.append(component_values)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            label = batch["label"].detach().cpu().numpy()
            probs.append(prob)
            labels.append(label)
    mean_loss = float(np.mean(losses)) if losses else math.nan
    component_summary: Dict[str, float] = {}
    if component_rows:
        component_df = pd.DataFrame(component_rows)
        component_summary = {
            key: float(component_df[key].mean())
            for key in component_df.columns
        }
    return mean_loss, np.concatenate(labels, axis=0), np.concatenate(probs, axis=0), component_summary


def infer_subject_paired_branch_prob(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    raw_probs: List[np.ndarray] = []
    biomarker_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            details = forward_subject_details_from_batch(model, batch)
            raw_probs.append(torch.sigmoid(details["raw_subject_logit"]).detach().cpu().numpy())
            biomarker_probs.append(torch.sigmoid(details["biomarker_logit"]).detach().cpu().numpy())
    return np.concatenate(raw_probs, axis=0), np.concatenate(biomarker_probs, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_from_batch(model, batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


def make_biomarker_bce_config(
    config: dict,
    positive_class_weight: float | str = 1.0,
    use_weighted_sampler: bool = False,
) -> dict:
    run_config = copy.deepcopy(config)
    run_config.setdefault("training", {})
    run_config["training"]["loss"] = {"name": "bce"}
    run_config["training"]["positive_class_weight"] = positive_class_weight
    run_config["training"]["use_weighted_sampler"] = bool(use_weighted_sampler)
    return run_config


def train_biomarker_mlp_with_early_stopping(
    config: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    train_labels: np.ndarray,
    input_dim: int,
    device: torch.device,
    seed: int,
    model_builder=None,
) -> Tuple[nn.Module, Dict[str, object]]:
    set_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stop_patience"])
    min_epochs = int(config["training"].get("min_epochs", 1))
    monitor_metric = str(config["training"].get("monitor_metric", "auprc")).lower()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not bool(config["training"].get("use_weighted_sampler", False)),
        sampler=(
            build_weighted_sampler(train_labels, seed=seed)
            if bool(config["training"].get("use_weighted_sampler", False))
            else None
        ),
        num_workers=int(config["training"]["num_workers"]),
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    if model_builder is None:
        model_builder = build_biomarker_mlp_model
    model = model_builder(config, input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion, loss_metadata = build_binary_classification_loss(config, train_labels, device)

    best_monitor = -math.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float]] = []
    threshold = float(config["training"].get("decision_threshold", 0.5))

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_y, val_prob = infer_sequence(model, val_loader, device, criterion=criterion)
        train_eval_loss, train_y, train_prob = infer_sequence(model, train_eval_loader, device, criterion=criterion)
        metrics = classification_metrics(val_y, val_prob, threshold=threshold)
        train_metrics = classification_metrics(train_y, train_prob, threshold=threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval_loss,
            "val_loss": val_loss,
        }
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in metrics.items():
            row[f"val_{key}"] = value
        history_rows.append(row)

        monitor_value = get_monitor_value(metrics, monitor_metric, val_loss)
        improved = np.isfinite(monitor_value) and (monitor_value > best_monitor)
        if improved:
            best_monitor = monitor_value
            best_epoch = epoch
            best_metrics = {"val_loss": val_loss, **metrics}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    return model, {
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "history": history_rows,
        "loss_metadata": loss_metadata,
    }


def train_biomarker_mlp_fixed_epochs(
    config: dict,
    train_ds: Dataset,
    train_labels: np.ndarray,
    input_dim: int,
    device: torch.device,
    seed: int,
    epochs: int,
    model_builder=None,
) -> Tuple[nn.Module, Dict[str, object]]:
    set_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not bool(config["training"].get("use_weighted_sampler", False)),
        sampler=(
            build_weighted_sampler(train_labels, seed=seed)
            if bool(config["training"].get("use_weighted_sampler", False))
            else None
        ),
        num_workers=int(config["training"]["num_workers"]),
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    if model_builder is None:
        model_builder = build_biomarker_mlp_model
    model = model_builder(config, input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion, loss_metadata = build_binary_classification_loss(config, train_labels, device)

    history_rows: List[Dict[str, float]] = []
    threshold = float(config["training"].get("decision_threshold", 0.5))
    for epoch in range(1, int(max(1, epochs)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_eval_loss, train_y, train_prob = infer_sequence(model, train_eval_loader, device, criterion=criterion)
        train_metrics = classification_metrics(train_y, train_prob, threshold=threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval_loss,
        }
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        history_rows.append(row)

    return model, {"history": history_rows, "loss_metadata": loss_metadata}


def train_one_epoch_subject_paired(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    aux_cfg: Dict[str, float | int | bool | None],
) -> Tuple[float, Dict[str, float]]:
    model.train()
    losses: List[float] = []
    component_rows: List[Dict[str, float]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, component_values = compute_subject_paired_loss(model, batch, criterion, aux_cfg)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        component_rows.append(component_values)

    mean_loss = float(np.mean(losses)) if losses else math.nan
    component_summary: Dict[str, float] = {}
    if component_rows:
        component_df = pd.DataFrame(component_rows)
        component_summary = {
            key: float(component_df[key].mean())
            for key in component_df.columns
        }
    return mean_loss, component_summary


def get_monitor_value(metrics: Dict[str, float], monitor_metric: str, val_loss: float) -> float:
    if monitor_metric == "loss":
        return -float(val_loss)
    candidate = float(metrics.get(monitor_metric, math.nan))
    if not np.isfinite(candidate):
        return -float(val_loss)
    return candidate


def run_baseline(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    df, feature_cols = load_aligned_tabular_features(config, manifest)

    fold_rows: List[Dict[str, float]] = []
    prediction_rows: List[Dict[str, float]] = []
    threshold = float(config["training"].get("decision_threshold", 0.5))

    for fold in sorted(df["fold"].dropna().unique()):
        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()
        if train_df.empty or val_df.empty:
            continue
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        model.fit(train_df[feature_cols], train_df["injury_label"])
        prob = model.predict_proba(val_df[feature_cols])[:, 1]
        metrics = classification_metrics(val_df["injury_label"].to_numpy(), prob, threshold=threshold)
        fold_row = {"fold": int(fold), "n_train": len(train_df), "n_val": len(val_df)}
        fold_row.update(metrics)
        fold_rows.append(fold_row)
        for _, row in val_df.iterrows():
            prediction_rows.append(
                {
                    "fold": int(fold),
                    "trial_id": row["trial_id"],
                    "subject_id": row["subject_id"],
                    "action": row["action"],
                    "label": float(row["injury_label"]),
                    "prob": float(prob[val_df.index.get_loc(row.name)]),
                }
            )

    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    fold_df.to_csv(reports_dir / "baseline_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(reports_dir / "baseline_val_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(reports_dir / "baseline_summary.json", summarize_fold_table(fold_df))
    return fold_df


def run_biomarker_mlp(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    logs_dir = outputs["logs"]
    ckpt_dir = outputs["checkpoints"] / "biomarker_mlp"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config)
    action_order = get_action_order(config)
    subject_df = build_subject_table(manifest, action_order)
    biomarker_df, biomarker_feature_cols, _, _ = load_subject_biomarker_features(config, subject_df)
    feature_tensor = torch.tensor(
        biomarker_df[biomarker_feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    label_tensor = torch.tensor(subject_df["injury_label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    write_json(
        reports_dir / "biomarker_mlp_features.json",
        {
            "n_features": len(biomarker_feature_cols),
            "feature_names": biomarker_feature_cols,
            "feature_package": str(config["data"]["biomarker_package_zip"]),
        },
    )

    threshold = float(config["training"].get("decision_threshold", 0.5))
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stop_patience"])
    min_epochs = int(config["training"].get("min_epochs", 1))
    monitor_metric = str(config["training"].get("monitor_metric", "auroc")).lower()
    use_weighted_sampler = bool(config["training"].get("use_weighted_sampler", False))
    random_seed = int(config["splits"].get("random_seed", 42))
    tuning_cfg = config.get("threshold_tuning", {})
    threshold_tuning_enabled = bool(tuning_cfg.get("enabled", False))
    threshold_strategies = [str(x).lower() for x in tuning_cfg.get("strategies", ["f1"])]
    primary_strategy = str(tuning_cfg.get("primary_strategy", threshold_strategies[0])).lower()
    if primary_strategy not in threshold_strategies:
        threshold_strategies = [primary_strategy] + [s for s in threshold_strategies if s != primary_strategy]

    fold_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    prediction_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    threshold_rows: List[Dict[str, float]] = []

    for fold in sorted(subject_df["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(subject_df["fold"].to_numpy() != fold)[0]
        val_idx = np.where(subject_df["fold"].to_numpy() == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        train_ds, val_ds, norm_state = normalize_subject_feature_fold_tensors(
            feature_tensor,
            label_tensor,
            train_idx,
            val_idx,
        )
        train_labels = label_tensor[train_idx].numpy().astype(int)
        train_sampler = None
        train_shuffle = True
        if use_weighted_sampler:
            train_sampler = build_weighted_sampler(train_labels, seed=random_seed + fold)
            train_shuffle = False

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=int(config["training"]["num_workers"]),
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )

        model = build_biomarker_mlp_model(config, input_dim=int(feature_tensor.shape[1])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        criterion, loss_metadata = build_binary_classification_loss(config, train_labels, device)

        best_monitor = -math.inf
        best_epoch = 0
        best_metrics: Dict[str, float] = {}
        epochs_without_improvement = 0
        history_rows: List[Dict[str, float]] = []

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_y, val_prob = infer_sequence(model, val_loader, device, criterion=criterion)
            metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            train_eval_loss, train_y, train_prob = infer_sequence(model, train_eval_loader, device, criterion=criterion)
            train_metrics = classification_metrics(train_y, train_prob, threshold=threshold)

            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
            }
            for key, value in train_metrics.items():
                row[f"train_{key}"] = value
            for key, value in metrics.items():
                row[f"val_{key}"] = value
            history_rows.append(row)

            monitor_value = get_monitor_value(metrics, monitor_metric, val_loss)
            improved = np.isfinite(monitor_value) and (monitor_value > best_monitor)
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_metrics = {"val_loss": val_loss, **metrics}
                checkpoint = {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_kwargs": {"input_dim": int(feature_tensor.shape[1])},
                    "feature_names": biomarker_feature_cols,
                    "loss_metadata": loss_metadata,
                    "normalization": {k: v.cpu() for k, v in norm_state.items()},
                    "best_metrics": best_metrics,
                    "config": config,
                }
                torch.save(checkpoint, ckpt_dir / f"fold_{fold}_best.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs and epochs_without_improvement >= patience:
                break

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(logs_dir / f"biomarker_mlp_fold_{fold}_history.csv", index=False, encoding="utf-8-sig")

        checkpoint = torch.load(ckpt_dir / f"fold_{fold}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, train_y_final, train_prob_final = infer_sequence(model, train_eval_loader, device, criterion=criterion)
        _, val_y, val_prob = infer_sequence(model, val_loader, device, criterion=criterion)

        val_subjects = subject_df.iloc[val_idx].reset_index(drop=True)

        if threshold_tuning_enabled:
            strategy_results = {}
            for strategy in threshold_strategies:
                tuned_threshold, train_score, train_metrics_tuned = tune_threshold(
                    train_y_final,
                    train_prob_final,
                    strategy=strategy,
                    config=config,
                )
                val_metrics = classification_metrics(val_y, val_prob, threshold=tuned_threshold)
                strategy_results[strategy] = {
                    "threshold": tuned_threshold,
                    "train_score": train_score,
                    "train_metrics": train_metrics_tuned,
                    "val_metrics": val_metrics,
                }
        else:
            default_metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            strategy_results = {
                primary_strategy: {
                    "threshold": threshold,
                    "train_score": math.nan,
                    "train_metrics": classification_metrics(train_y_final, train_prob_final, threshold=threshold),
                    "val_metrics": default_metrics,
                }
            }

        for strategy, result in strategy_results.items():
            threshold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "threshold": float(result["threshold"]),
                    "train_objective": float(result["train_score"]),
                    "train_f1": float(result["train_metrics"]["f1"]),
                    "train_youden": float(youden_index(result["train_metrics"])),
                    "val_f1": float(result["val_metrics"]["f1"]),
                    "val_youden": float(youden_index(result["val_metrics"])),
                }
            )

            fold_row = {
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "best_epoch": int(best_epoch),
                "device": str(device),
                "pos_weight": float(loss_metadata.get("loss_pos_weight", math.nan)),
                "weighted_sampler": int(use_weighted_sampler),
                "loss_name": str(loss_metadata.get("loss_name", "unknown")),
                "loss_gamma": float(loss_metadata.get("loss_gamma", math.nan)),
                "loss_neg_weight": float(loss_metadata.get("loss_neg_weight", math.nan)),
                "threshold_strategy": strategy,
                "threshold": float(result["threshold"]),
                "train_threshold_objective": float(result["train_score"]),
            }
            if "loss_cb_beta" in loss_metadata:
                fold_row["loss_cb_beta"] = float(loss_metadata["loss_cb_beta"])
            fold_row.update(result["val_metrics"])
            fold_row["val_loss"] = float(best_metrics.get("val_loss", math.nan))
            fold_rows_by_strategy[strategy].append(fold_row)

            for i, row in val_subjects.iterrows():
                prediction_rows_by_strategy[strategy].append(
                    {
                        "fold": fold,
                        "threshold_strategy": strategy,
                        "threshold": float(result["threshold"]),
                        "subject_id": row["subject_id"],
                        "label": float(row["injury_label"]),
                        "prob": float(val_prob[i]),
                        "pred_label": int(float(val_prob[i]) >= float(result["threshold"])),
                    }
                )

    primary_fold_df = None
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(reports_dir / "biomarker_mlp_threshold_tuning.csv", index=False, encoding="utf-8-sig")

    for strategy, rows in fold_rows_by_strategy.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(prediction_rows_by_strategy[strategy])
        suffix = "" if strategy == primary_strategy else f"_{strategy}"
        fold_df.to_csv(reports_dir / f"biomarker_mlp_fold_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(reports_dir / f"biomarker_mlp_val_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
        experiment_summary = {
            "device": str(device),
            "n_trials": int(len(manifest)),
            "n_subjects": int(len(subject_df)),
            "feature_count": int(len(biomarker_feature_cols)),
            "threshold_strategy": strategy,
            "fold_summary": summarize_fold_table(fold_df),
        }
        write_json(reports_dir / f"biomarker_mlp_summary{suffix}.json", experiment_summary)
        if strategy == primary_strategy:
            primary_fold_df = fold_df

    if primary_fold_df is None:
        raise RuntimeError("primary threshold strategy results were not generated")
    return primary_fold_df


def run_biomarker_mlp_nested(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    logs_dir = outputs["logs"]

    nested_cfg = config.get("biomarker_mlp_nested", {})
    nested_pos_weight = nested_cfg.get("positive_class_weight", 1.0)
    nested_use_weighted_sampler = bool(nested_cfg.get("use_weighted_sampler", False))
    ensemble_size = int(nested_cfg.get("ensemble_size", 1))
    ensemble_size = max(1, ensemble_size)
    ensemble_seed_stride = int(nested_cfg.get("ensemble_seed_stride", 10000))
    report_prefix = (
        "biomarker_mlp_nested_bce_calibrated"
        if ensemble_size == 1
        else f"biomarker_mlp_nested_bce_calibrated_ensemble{ensemble_size}"
    )
    ckpt_dir = outputs["checkpoints"] / report_prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_config = make_biomarker_bce_config(
        config,
        positive_class_weight=nested_pos_weight,
        use_weighted_sampler=nested_use_weighted_sampler,
    )
    inner_splits_requested = int(nested_cfg.get("inner_splits", 4))
    calibration_enabled = bool(nested_cfg.get("calibration_enabled", True))
    calibration_c = float(nested_cfg.get("calibration_c", 1.0))
    final_epoch_aggregation = str(nested_cfg.get("final_epoch_aggregation", "median")).lower()
    min_final_epochs = int(nested_cfg.get("min_final_epochs", 1))
    max_final_epochs = int(nested_cfg.get("max_final_epochs", run_config["training"]["max_epochs"]))
    threshold_strategies = [
        str(x).lower()
        for x in nested_cfg.get("strategies", ["accuracy", "balanced_accuracy", "f1", "youden"])
    ]
    primary_strategy = str(nested_cfg.get("primary_strategy", threshold_strategies[0])).lower()
    if primary_strategy not in threshold_strategies:
        threshold_strategies = [primary_strategy] + [s for s in threshold_strategies if s != primary_strategy]

    device = choose_device(run_config)
    action_order = get_action_order(run_config)
    subject_df = build_subject_table(manifest, action_order)
    biomarker_df, biomarker_feature_cols, _, _ = load_subject_biomarker_features(run_config, subject_df)
    feature_tensor = torch.tensor(
        biomarker_df[biomarker_feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    label_tensor = torch.tensor(subject_df["injury_label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    write_json(
        reports_dir / f"{report_prefix}_features.json",
        {
            "n_features": len(biomarker_feature_cols),
            "feature_names": biomarker_feature_cols,
            "feature_package": str(run_config["data"]["biomarker_package_zip"]),
            "loss": "bce",
            "positive_class_weight": nested_pos_weight,
            "use_weighted_sampler": nested_use_weighted_sampler,
            "ensemble_size": ensemble_size,
            "ensemble_seed_stride": ensemble_seed_stride,
            "calibration": "platt" if calibration_enabled else "identity",
            "inner_splits_requested": inner_splits_requested,
        },
    )

    batch_size = int(run_config["training"]["batch_size"])
    random_seed = int(run_config["splits"].get("random_seed", 42))
    input_dim = int(feature_tensor.shape[1])

    fold_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    prediction_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    threshold_rows: List[Dict[str, float]] = []
    inner_fold_rows: List[Dict[str, float]] = []
    inner_oof_rows: List[Dict[str, float]] = []
    final_history_rows: List[Dict[str, float]] = []

    for fold in sorted(subject_df["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(subject_df["fold"].to_numpy() != fold)[0]
        val_idx = np.where(subject_df["fold"].to_numpy() == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        outer_train_labels = label_tensor[train_idx].numpy().astype(int)
        class_counts = np.bincount(outer_train_labels, minlength=2)
        inner_splits = int(min(inner_splits_requested, class_counts.min()))
        if inner_splits < 2:
            raise RuntimeError(f"outer fold {fold} has too few minority samples for nested OOF thresholding")

        oof_logits = np.full(train_idx.size, np.nan, dtype=np.float64)
        inner_best_epochs: List[int] = []
        splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed + 1000 + fold)

        for inner_id, (inner_train_rel, inner_val_rel) in enumerate(splitter.split(np.zeros(train_idx.size), outer_train_labels)):
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]
            train_ds, val_ds, _ = normalize_subject_feature_fold_tensors(
                feature_tensor,
                label_tensor,
                inner_train_idx,
                inner_val_idx,
            )
            criterion, loss_metadata = build_binary_classification_loss(
                run_config,
                label_tensor[inner_train_idx].numpy().astype(int),
                device,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(run_config["training"]["num_workers"]),
            )
            inner_logits_by_seed: List[np.ndarray] = []
            inner_losses_by_seed: List[float] = []
            inner_seed_epochs: List[int] = []
            inner_y = np.asarray([], dtype=np.float32)
            for ensemble_id in range(ensemble_size):
                inner_seed = random_seed + fold * 10007 + inner_id * 1009 + ensemble_id * ensemble_seed_stride
                inner_model, inner_info = train_biomarker_mlp_with_early_stopping(
                    run_config,
                    train_ds,
                    val_ds,
                    label_tensor[inner_train_idx].numpy().astype(int),
                    input_dim=input_dim,
                    device=device,
                    seed=inner_seed,
                )
                inner_loss, inner_y, _, inner_logits_one = infer_sequence_with_logits(
                    inner_model,
                    val_loader,
                    device,
                    criterion=criterion,
                )
                inner_logits_by_seed.append(inner_logits_one.astype(np.float64))
                inner_losses_by_seed.append(float(inner_loss))
                inner_seed_epochs.append(int(inner_info.get("best_epoch", 0)))

            inner_logits_stack = np.stack(inner_logits_by_seed, axis=0)
            inner_logits = inner_logits_stack.mean(axis=0)
            inner_logit_std = inner_logits_stack.std(axis=0)
            inner_prob = sigmoid_numpy(inner_logits)
            oof_logits[inner_val_rel] = inner_logits.astype(np.float64)
            inner_best_epochs.extend(inner_seed_epochs)
            inner_metrics = classification_metrics(inner_y, inner_prob, threshold=0.5)
            inner_fold_row = {
                "outer_fold": fold,
                "inner_fold": int(inner_id),
                "ensemble_size": int(ensemble_size),
                "n_train": int(inner_train_idx.size),
                "n_val": int(inner_val_idx.size),
                "best_epoch": int(round(float(np.median(inner_seed_epochs)))),
                "best_epoch_mean": float(np.mean(inner_seed_epochs)),
                "best_epoch_min": int(np.min(inner_seed_epochs)),
                "best_epoch_max": int(np.max(inner_seed_epochs)),
                "val_loss": float(np.mean(inner_losses_by_seed)),
                "loss_name": str(loss_metadata.get("loss_name", "unknown")),
                "pos_weight": float(loss_metadata.get("loss_pos_weight", math.nan)),
                "threshold": 0.5,
            }
            inner_fold_row.update(inner_metrics)
            inner_fold_rows.append(inner_fold_row)

            inner_subjects = subject_df.iloc[inner_val_idx].reset_index(drop=True)
            for i, row in inner_subjects.iterrows():
                inner_oof_rows.append(
                    {
                        "outer_fold": fold,
                        "inner_fold": int(inner_id),
                        "subject_id": row["subject_id"],
                        "label": float(row["injury_label"]),
                        "oof_logit_uncalibrated": float(inner_logits[i]),
                        "oof_logit_std": float(inner_logit_std[i]),
                        "oof_prob_uncalibrated": float(inner_prob[i]),
                    }
                )

        if not np.isfinite(oof_logits).all():
            raise RuntimeError(f"outer fold {fold} did not produce complete OOF logits")

        calibrator = fit_platt_calibrator(
            oof_logits,
            outer_train_labels,
            c_value=calibration_c,
            enabled=calibration_enabled,
        )
        oof_prob_uncalibrated = sigmoid_numpy(oof_logits)
        oof_prob_calibrated = apply_platt_calibrator(oof_logits, calibrator)
        oof_metrics_uncalibrated = classification_metrics(outer_train_labels, oof_prob_uncalibrated, threshold=0.5)
        oof_metrics_calibrated = classification_metrics(outer_train_labels, oof_prob_calibrated, threshold=0.5)

        if final_epoch_aggregation == "mean":
            final_epochs = int(round(float(np.mean(inner_best_epochs))))
        elif final_epoch_aggregation == "max":
            final_epochs = int(max(inner_best_epochs))
        else:
            final_epochs = int(round(float(np.median(inner_best_epochs))))
        final_epochs = int(np.clip(final_epochs, min_final_epochs, max_final_epochs))

        train_ds, val_ds, norm_state = normalize_subject_feature_fold_tensors(
            feature_tensor,
            label_tensor,
            train_idx,
            val_idx,
        )
        final_criterion, final_loss_metadata = build_binary_classification_loss(run_config, outer_train_labels, device)
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(run_config["training"]["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(run_config["training"]["num_workers"]),
        )
        train_logits_by_seed: List[np.ndarray] = []
        val_logits_by_seed: List[np.ndarray] = []
        val_losses_by_seed: List[float] = []
        final_model_states: List[Dict[str, torch.Tensor]] = []
        train_y_final = np.asarray([], dtype=np.float32)
        val_y = np.asarray([], dtype=np.float32)
        for ensemble_id in range(ensemble_size):
            final_seed = random_seed + fold * 997 + ensemble_id * ensemble_seed_stride
            final_model, final_info = train_biomarker_mlp_fixed_epochs(
                run_config,
                train_ds,
                outer_train_labels,
                input_dim=input_dim,
                device=device,
                seed=final_seed,
                epochs=final_epochs,
            )
            _, train_y_final, _, train_logits_one = infer_sequence_with_logits(
                final_model,
                train_eval_loader,
                device,
                criterion=final_criterion,
            )
            val_loss_one, val_y, _, val_logits_one = infer_sequence_with_logits(
                final_model,
                val_loader,
                device,
                criterion=final_criterion,
            )
            train_logits_by_seed.append(train_logits_one.astype(np.float64))
            val_logits_by_seed.append(val_logits_one.astype(np.float64))
            val_losses_by_seed.append(float(val_loss_one))
            final_model_states.append(
                {key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()}
            )
            for row in final_info.get("history", []):
                final_history_rows.append({"fold": fold, "ensemble_id": ensemble_id, "seed": final_seed, **row})

        train_logits_stack = np.stack(train_logits_by_seed, axis=0)
        val_logits_stack = np.stack(val_logits_by_seed, axis=0)
        train_logits_uncalibrated = train_logits_stack.mean(axis=0)
        val_logits_uncalibrated = val_logits_stack.mean(axis=0)
        val_logit_std = val_logits_stack.std(axis=0)
        train_prob_uncalibrated = sigmoid_numpy(train_logits_uncalibrated)
        val_prob_uncalibrated = sigmoid_numpy(val_logits_uncalibrated)
        val_loss = float(np.mean(val_losses_by_seed))
        train_prob_calibrated = apply_platt_calibrator(train_logits_uncalibrated, calibrator)
        val_prob_calibrated = apply_platt_calibrator(val_logits_uncalibrated, calibrator)
        val_uncalibrated_metrics = classification_metrics(val_y, val_prob_uncalibrated, threshold=0.5)

        strategy_results = {}
        for strategy in threshold_strategies:
            tuned_threshold, train_score, train_metrics_tuned = tune_threshold(
                outer_train_labels,
                oof_prob_calibrated,
                strategy=strategy,
                config=run_config,
            )
            val_metrics = classification_metrics(val_y, val_prob_calibrated, threshold=tuned_threshold)
            strategy_results[strategy] = {
                "threshold": tuned_threshold,
                "train_score": train_score,
                "train_metrics": train_metrics_tuned,
                "val_metrics": val_metrics,
            }

        checkpoint = {
            "fold": fold,
            "model_state_dicts": final_model_states,
            "model_kwargs": {"input_dim": input_dim},
            "feature_names": biomarker_feature_cols,
            "loss_metadata": final_loss_metadata,
            "normalization": {k: v.cpu() for k, v in norm_state.items()},
            "calibrator": calibrator,
            "thresholds": {
                strategy: float(result["threshold"])
                for strategy, result in strategy_results.items()
            },
            "config": run_config,
            "final_epochs": final_epochs,
            "ensemble_size": ensemble_size,
        }
        torch.save(checkpoint, ckpt_dir / f"fold_{fold}_final.pt")

        val_subjects = subject_df.iloc[val_idx].reset_index(drop=True)
        for strategy, result in strategy_results.items():
            threshold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "threshold": float(result["threshold"]),
                    "oof_objective": float(result["train_score"]),
                    "oof_f1": float(result["train_metrics"]["f1"]),
                    "oof_youden": float(youden_index(result["train_metrics"])),
                    "oof_accuracy": float(result["train_metrics"]["accuracy"]),
                    "val_f1": float(result["val_metrics"]["f1"]),
                    "val_youden": float(youden_index(result["val_metrics"])),
                    "val_accuracy": float(result["val_metrics"]["accuracy"]),
                }
            )

            fold_row = {
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "ensemble_size": int(ensemble_size),
                "inner_splits": int(inner_splits),
                "inner_best_epoch_mean": float(np.mean(inner_best_epochs)),
                "inner_best_epoch_median": float(np.median(inner_best_epochs)),
                "final_epochs": int(final_epochs),
                "device": str(device),
                "pos_weight": float(final_loss_metadata.get("loss_pos_weight", math.nan)),
                "weighted_sampler": int(nested_use_weighted_sampler),
                "loss_name": str(final_loss_metadata.get("loss_name", "unknown")),
                "loss_gamma": float(final_loss_metadata.get("loss_gamma", math.nan)),
                "loss_neg_weight": float(final_loss_metadata.get("loss_neg_weight", math.nan)),
                "calibration_method": str(calibrator.get("method", "identity")),
                "calibration_coef": float(calibrator.get("coef", math.nan)),
                "calibration_intercept": float(calibrator.get("intercept", math.nan)),
                "threshold_strategy": strategy,
                "threshold": float(result["threshold"]),
                "train_threshold_objective": float(result["train_score"]),
                "oof_auroc_uncalibrated": float(oof_metrics_uncalibrated["auroc"]),
                "oof_auprc_uncalibrated": float(oof_metrics_uncalibrated["auprc"]),
                "oof_auroc_calibrated": float(oof_metrics_calibrated["auroc"]),
                "oof_auprc_calibrated": float(oof_metrics_calibrated["auprc"]),
                "val_auroc_uncalibrated": float(val_uncalibrated_metrics["auroc"]),
                "val_auprc_uncalibrated": float(val_uncalibrated_metrics["auprc"]),
                "val_loss": float(val_loss),
            }
            fold_row.update(result["val_metrics"])
            fold_rows_by_strategy[strategy].append(fold_row)

            for i, row in val_subjects.iterrows():
                prediction_rows_by_strategy[strategy].append(
                    {
                        "fold": fold,
                        "threshold_strategy": strategy,
                        "threshold": float(result["threshold"]),
                        "subject_id": row["subject_id"],
                        "label": float(row["injury_label"]),
                        "prob": float(val_prob_calibrated[i]),
                        "prob_uncalibrated": float(val_prob_uncalibrated[i]),
                        "logit_uncalibrated": float(val_logits_uncalibrated[i]),
                        "logit_std": float(val_logit_std[i]),
                        "pred_label": int(float(val_prob_calibrated[i]) >= float(result["threshold"])),
                    }
                )

        for idx, row in enumerate(inner_oof_rows):
            if int(row["outer_fold"]) == fold:
                rel = np.where(subject_df.iloc[train_idx]["subject_id"].astype(str).to_numpy() == str(row["subject_id"]))[0]
                if rel.size > 0:
                    row["oof_prob"] = float(oof_prob_calibrated[int(rel[0])])

    pd.DataFrame(inner_fold_rows).to_csv(
        reports_dir / f"{report_prefix}_inner_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(inner_oof_rows).to_csv(
        reports_dir / f"{report_prefix}_inner_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(final_history_rows).to_csv(
        logs_dir / f"{report_prefix}_final_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(reports_dir / f"{report_prefix}_threshold_tuning.csv", index=False, encoding="utf-8-sig")

    primary_fold_df = None
    for strategy, rows in fold_rows_by_strategy.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(prediction_rows_by_strategy[strategy])
        suffix = "" if strategy == primary_strategy else f"_{strategy}"
        fold_df.to_csv(reports_dir / f"{report_prefix}_fold_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(reports_dir / f"{report_prefix}_val_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
        experiment_summary = {
            "device": str(device),
            "n_trials": int(len(manifest)),
            "n_subjects": int(len(subject_df)),
            "feature_count": int(len(biomarker_feature_cols)),
            "loss": "bce",
            "positive_class_weight": nested_pos_weight,
            "weighted_sampler": bool(nested_use_weighted_sampler),
            "ensemble_size": int(ensemble_size),
            "calibration": "platt" if calibration_enabled else "identity",
            "nested_threshold": True,
            "threshold_strategy": strategy,
            "fold_summary": summarize_fold_table(fold_df),
        }
        write_json(reports_dir / f"{report_prefix}_summary{suffix}.json", experiment_summary)
        if strategy == primary_strategy:
            primary_fold_df = fold_df

    if primary_fold_df is None:
        raise RuntimeError("primary nested biomarker MLP results were not generated")
    return primary_fold_df


def bootstrap_threshold_distribution(
    y_true: np.ndarray,
    prob: np.ndarray,
    strategy: str,
    config: dict,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    finite_mask = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[finite_mask]
    prob = prob[finite_mask]
    if y_true.size == 0:
        raise ValueError("cannot bootstrap thresholds from an empty OOF set")

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    thresholds: List[float] = []
    scores: List[float] = []
    n_bootstrap = int(max(1, n_bootstrap))
    for _ in range(n_bootstrap):
        if pos_idx.size > 0 and neg_idx.size > 0:
            sample_idx = np.concatenate(
                [
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                ]
            )
            rng.shuffle(sample_idx)
        else:
            sample_idx = rng.choice(np.arange(y_true.size), size=y_true.size, replace=True)
        threshold, score, _ = tune_threshold_fast(
            y_true[sample_idx],
            prob[sample_idx],
            strategy=strategy,
            config=config,
        )
        thresholds.append(float(threshold))
        scores.append(float(score))
    return np.asarray(thresholds, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def add_metric_columns(row: Dict[str, object], metrics: Dict[str, float], prefix: str = "") -> None:
    for key, value in metrics.items():
        row[f"{prefix}{key}"] = safe_float(float(value))


def compact_feature_list(rows: List[Dict[str, object]], limit: int = 5) -> str:
    items = []
    for row in rows[:limit]:
        items.append(
            f"{row['feature']}(score={float(row['positive_side_score']):.3f},"
            f"z={float(row['zscore']):.2f},pct={float(row['train_percentile']):.2f})"
        )
    return ";".join(items)


def compact_neighbor_list(rows: List[Dict[str, object]], limit: int = 5) -> str:
    items = []
    for row in rows[:limit]:
        prob = row.get("neighbor_oof_prob")
        prob_text = "nan" if prob is None or not np.isfinite(float(prob)) else f"{float(prob):.3f}"
        items.append(
            f"{row['neighbor_subject_id']}(y={int(row['neighbor_label'])},"
            f"d={float(row['distance']):.2f},oof={prob_text})"
        )
    return ";".join(items)


def summarize_threshold_bagging(
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    group_cols = ["bootstrap_objective", "threshold_aggregation"]
    metric_cols = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1", "tp", "tn", "fp", "fn"]

    for keys, group in fold_df.groupby(group_cols, sort=False):
        objective, aggregation = keys
        pred_group = pred_df[
            (pred_df["bootstrap_objective"] == objective)
            & (pred_df["threshold_aggregation"] == aggregation)
        ].copy()
        pooled = classification_metrics_from_predictions(
            pred_group["label"].to_numpy(),
            pred_group["prob"].to_numpy(),
            pred_group["pred_label"].to_numpy(),
        )
        row: Dict[str, object] = {
            "bootstrap_objective": objective,
            "threshold_aggregation": aggregation,
            "n_folds": int(group["fold"].nunique()),
            "threshold_mean": safe_float(pd.to_numeric(group["threshold"], errors="coerce").mean()),
            "threshold_std": safe_float(pd.to_numeric(group["threshold"], errors="coerce").std(ddof=1)),
            "threshold_min": safe_float(pd.to_numeric(group["threshold"], errors="coerce").min()),
            "threshold_max": safe_float(pd.to_numeric(group["threshold"], errors="coerce").max()),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"fold_mean_{col}"] = safe_float(values.mean())
            row[f"fold_std_{col}"] = safe_float(values.std(ddof=1))
        for col, value in pooled.items():
            row[f"pooled_{col}"] = safe_float(float(value))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_json = {
        "strategy_summary": summary_rows,
        "fold_summary": {
            f"{objective}_{aggregation}": summarize_fold_table(group.drop(columns=["bootstrap_objective", "threshold_aggregation"]))
            for (objective, aggregation), group in fold_df.groupby(group_cols, sort=False)
        },
    }
    return summary_df, summary_json


def generate_threshold_bagging_error_reports(
    config: dict,
    manifest: pd.DataFrame,
    pred_df: pd.DataFrame,
    source_pred_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    top_features: int,
    neighbor_k: int,
    boundary_margin: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_order = get_action_order(config)
    subject_df = build_subject_table(manifest, action_order)
    subject_df["subject_id"] = subject_df["subject_id"].map(normalize_subject_id)
    biomarker_df, feature_cols, _, _ = load_subject_biomarker_features(config, subject_df)
    x_all = biomarker_df[feature_cols].to_numpy(dtype=np.float64)
    label_all = subject_df["injury_label"].to_numpy(dtype=int)
    subject_ids = subject_df["subject_id"].map(normalize_subject_id).to_numpy()
    fold_all = subject_df["fold"].to_numpy(dtype=int)

    source_lookup = source_pred_df.copy()
    source_lookup["subject_id"] = source_lookup["subject_id"].map(normalize_subject_id)
    source_lookup = source_lookup.set_index("subject_id", drop=False)

    oof_lookup = oof_df.copy()
    oof_lookup["subject_id"] = oof_lookup["subject_id"].map(normalize_subject_id)

    subject_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    neighbor_rows: List[Dict[str, object]] = []

    for fold in sorted(pred_df["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(fold_all != fold)[0]
        val_idx = np.where(fold_all == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train_raw = imputer.fit_transform(x_all[train_idx])
        x_val_raw = imputer.transform(x_all[val_idx])
        x_train_raw = np.nan_to_num(x_train_raw, nan=0.0).astype(np.float64, copy=False)
        x_val_raw = np.nan_to_num(x_val_raw, nan=0.0).astype(np.float64, copy=False)
        x_train_z = scaler.fit_transform(x_train_raw).astype(np.float64, copy=False)
        x_val_z = scaler.transform(x_val_raw).astype(np.float64, copy=False)
        train_y = label_all[train_idx].astype(int)

        class0_mean = (
            x_train_z[train_y == 0].mean(axis=0)
            if np.any(train_y == 0)
            else np.full(len(feature_cols), np.nan)
        )
        class1_mean = (
            x_train_z[train_y == 1].mean(axis=0)
            if np.any(train_y == 1)
            else np.full(len(feature_cols), np.nan)
        )
        class_mid = 0.5 * (class0_mean + class1_mean)
        class_contrast = class1_mean - class0_mean
        contrast_sign = np.sign(class_contrast)
        contrast_sign[contrast_sign == 0] = 1.0

        train_subject_ids = subject_ids[train_idx]
        val_subject_ids = subject_ids[val_idx]
        train_oof = oof_lookup[oof_lookup["outer_fold"].astype(int) == fold].copy()
        train_oof_prob = {
            normalize_subject_id(row["subject_id"]): float(row["oof_prob"])
            for _, row in train_oof.iterrows()
            if "oof_prob" in row and pd.notna(row["oof_prob"])
        }

        fold_pred = pred_df[pred_df["fold"].astype(int) == fold].copy()
        fold_errors = fold_pred[fold_pred["pred_label"].astype(int) != fold_pred["label"].astype(int)]
        for _, pred_row in fold_errors.iterrows():
            subject_id = normalize_subject_id(pred_row["subject_id"])
            local_matches = np.where(val_subject_ids == subject_id)[0]
            if local_matches.size == 0:
                continue
            local_idx = int(local_matches[0])
            label = int(pred_row["label"])
            pred_label = int(pred_row["pred_label"])
            error_type = "FP" if label == 0 and pred_label == 1 else "FN"
            margin = float(pred_row["prob"]) - float(pred_row["threshold"])
            source_info = source_lookup.loc[subject_id] if subject_id in source_lookup.index else None

            distances = np.linalg.norm(x_train_z - x_val_z[local_idx][None, :], axis=1)
            order = np.argsort(distances)
            neighbor_candidates: List[Dict[str, object]] = []
            for rank, train_pos in enumerate(order[: max(neighbor_k, 1) * 4], start=1):
                neighbor_subject_id = str(train_subject_ids[train_pos])
                neighbor_row = {
                    "bootstrap_objective": pred_row["bootstrap_objective"],
                    "threshold_aggregation": pred_row["threshold_aggregation"],
                    "fold": fold,
                    "subject_id": subject_id,
                    "label": label,
                    "pred_label": pred_label,
                    "error_type": error_type,
                    "rank": rank,
                    "neighbor_subject_id": neighbor_subject_id,
                    "neighbor_label": int(train_y[train_pos]),
                    "distance": float(distances[train_pos]),
                    "neighbor_oof_prob": safe_float(train_oof_prob.get(neighbor_subject_id, math.nan)),
                }
                neighbor_rows.append(neighbor_row)
                neighbor_candidates.append(neighbor_row)
                if rank >= neighbor_k:
                    continue

            row_feature_details: List[Dict[str, object]] = []
            for feature_idx, feature_name in enumerate(feature_cols):
                train_values = x_train_raw[:, feature_idx]
                value = float(x_val_raw[local_idx, feature_idx])
                percentile = float(np.mean(train_values <= value))
                zscore = float(x_val_z[local_idx, feature_idx])
                positive_side_score = float((zscore - class_mid[feature_idx]) * contrast_sign[feature_idx])
                feature_row: Dict[str, object] = {
                    "bootstrap_objective": pred_row["bootstrap_objective"],
                    "threshold_aggregation": pred_row["threshold_aggregation"],
                    "fold": fold,
                    "subject_id": subject_id,
                    "label": label,
                    "pred_label": pred_label,
                    "error_type": error_type,
                    "prob": float(pred_row["prob"]),
                    "threshold": float(pred_row["threshold"]),
                    "margin": margin,
                    "feature": feature_name,
                    "raw_value_imputed": value,
                    "zscore": zscore,
                    "train_percentile": percentile,
                    "class0_mean_z": safe_float(class0_mean[feature_idx]),
                    "class1_mean_z": safe_float(class1_mean[feature_idx]),
                    "class_contrast_z": safe_float(class_contrast[feature_idx]),
                    "positive_side_score": safe_float(positive_side_score),
                    "direction": "positive_like" if positive_side_score >= 0 else "negative_like",
                    "supports_error": bool(
                        (error_type == "FP" and positive_side_score > 0)
                        or (error_type == "FN" and positive_side_score < 0)
                    ),
                }
                feature_rows.append(feature_row)
                row_feature_details.append(feature_row)

            positive_like = sorted(
                row_feature_details,
                key=lambda item: float(item["positive_side_score"]),
                reverse=True,
            )
            negative_like = sorted(
                row_feature_details,
                key=lambda item: float(item["positive_side_score"]),
            )
            error_support = positive_like if error_type == "FP" else negative_like
            counter_evidence = negative_like if error_type == "FP" else positive_like
            same_label_neighbors = [row for row in neighbor_candidates if int(row["neighbor_label"]) == label]
            opposite_label_neighbors = [row for row in neighbor_candidates if int(row["neighbor_label"]) != label]

            subject_row: Dict[str, object] = {
                "bootstrap_objective": pred_row["bootstrap_objective"],
                "threshold_aggregation": pred_row["threshold_aggregation"],
                "fold": fold,
                "subject_id": subject_id,
                "label": label,
                "pred_label": pred_label,
                "error_type": error_type,
                "prob": float(pred_row["prob"]),
                "threshold": float(pred_row["threshold"]),
                "margin": margin,
                "abs_margin": abs(margin),
                "near_boundary": bool(abs(margin) <= boundary_margin),
                "prob_uncalibrated": safe_float(pred_row.get("prob_uncalibrated", math.nan)),
                "logit_uncalibrated": safe_float(pred_row.get("logit_uncalibrated", math.nan)),
                "logit_std": safe_float(pred_row.get("logit_std", math.nan)),
                "source_accuracy_threshold": safe_float(source_info["threshold"]) if source_info is not None else None,
                "source_accuracy_pred_label": int(source_info["pred_label"]) if source_info is not None else None,
                "source_accuracy_margin": (
                    safe_float(float(source_info["prob"]) - float(source_info["threshold"]))
                    if source_info is not None
                    else None
                ),
                "top_error_support_features": compact_feature_list(error_support, limit=top_features),
                "top_counter_features": compact_feature_list(counter_evidence, limit=top_features),
                "nearest_neighbors": compact_neighbor_list(neighbor_candidates, limit=neighbor_k),
                "nearest_same_label_neighbors": compact_neighbor_list(same_label_neighbors, limit=neighbor_k),
                "nearest_opposite_label_neighbors": compact_neighbor_list(opposite_label_neighbors, limit=neighbor_k),
            }
            subject_rows.append(subject_row)

    return pd.DataFrame(subject_rows), pd.DataFrame(feature_rows), pd.DataFrame(neighbor_rows)


def run_threshold_bagging(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    bag_cfg = config.get("threshold_bagging", {})
    source_prefix = str(
        bag_cfg.get("source_prefix", "biomarker_mlp_nested_bce_calibrated_ensemble10")
    )
    report_prefix = str(
        bag_cfg.get("report_prefix", f"{source_prefix}_threshold_bagging")
    )
    val_path = reports_dir / f"{source_prefix}_val_predictions.csv"
    oof_path = reports_dir / f"{source_prefix}_inner_oof_predictions.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"source validation predictions not found: {val_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"source inner OOF predictions not found: {oof_path}")

    source_pred_df = pd.read_csv(val_path)
    oof_df = pd.read_csv(oof_path)
    for df in (source_pred_df, oof_df):
        df["subject_id"] = df["subject_id"].map(normalize_subject_id)

    if "oof_prob" not in oof_df.columns:
        raise ValueError(f"{oof_path} must contain calibrated oof_prob")
    if "prob" not in source_pred_df.columns:
        raise ValueError(f"{val_path} must contain calibrated prob")

    n_bootstrap = int(bag_cfg.get("n_bootstrap", 1000))
    objectives = [str(x).lower() for x in bag_cfg.get("objectives", ["accuracy"])]
    aggregations = bag_cfg.get(
        "aggregations",
        {"median": 0.50, "q65": 0.65, "q75": 0.75},
    )
    if isinstance(aggregations, list):
        aggregation_map = {}
        for name in aggregations:
            key = str(name).lower()
            aggregation_map[key] = 0.50 if key == "median" else float(key.lstrip("q")) / 100.0
    else:
        aggregation_map = {str(name).lower(): float(q) for name, q in dict(aggregations).items()}
    random_seed = int(bag_cfg.get("random_seed", int(config["splits"].get("random_seed", 42)) + 2026))
    top_features = int(bag_cfg.get("top_features", 5))
    neighbor_k = int(bag_cfg.get("neighbor_k", 5))
    boundary_margin = float(bag_cfg.get("boundary_margin", 0.02))

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    threshold_sample_rows: List[Dict[str, object]] = []

    for fold in sorted(source_pred_df["fold"].dropna().unique()):
        fold = int(fold)
        fold_oof = oof_df[oof_df["outer_fold"].astype(int) == fold].copy()
        fold_val = source_pred_df[source_pred_df["fold"].astype(int) == fold].copy()
        if fold_oof.empty or fold_val.empty:
            continue

        y_oof = fold_oof["label"].to_numpy(dtype=int)
        prob_oof = fold_oof["oof_prob"].to_numpy(dtype=np.float64)
        y_val = fold_val["label"].to_numpy(dtype=int)
        prob_val = fold_val["prob"].to_numpy(dtype=np.float64)

        for objective in objectives:
            rng = np.random.default_rng(random_seed + fold * 1009 + sum(ord(ch) for ch in objective))
            thresholds, scores = bootstrap_threshold_distribution(
                y_oof,
                prob_oof,
                strategy=objective,
                config=config,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            for bootstrap_id, (threshold, score) in enumerate(zip(thresholds, scores)):
                threshold_sample_rows.append(
                    {
                        "fold": fold,
                        "bootstrap_objective": objective,
                        "bootstrap_id": int(bootstrap_id),
                        "threshold": float(threshold),
                        "objective_score": safe_float(float(score)),
                    }
                )

            full_oof_threshold, full_oof_score, full_oof_metrics = tune_threshold_fast(
                y_oof,
                prob_oof,
                strategy=objective,
                config=config,
            )
            threshold_base_row: Dict[str, object] = {
                "fold": fold,
                "bootstrap_objective": objective,
                "n_bootstrap": int(n_bootstrap),
                "full_oof_threshold": float(full_oof_threshold),
                "full_oof_objective_score": safe_float(float(full_oof_score)),
                "bootstrap_threshold_mean": safe_float(float(np.mean(thresholds))),
                "bootstrap_threshold_std": safe_float(float(np.std(thresholds, ddof=1))),
                "bootstrap_threshold_min": safe_float(float(np.min(thresholds))),
                "bootstrap_threshold_q25": safe_float(float(np.quantile(thresholds, 0.25))),
                "bootstrap_threshold_median": safe_float(float(np.quantile(thresholds, 0.50))),
                "bootstrap_threshold_q65": safe_float(float(np.quantile(thresholds, 0.65))),
                "bootstrap_threshold_q75": safe_float(float(np.quantile(thresholds, 0.75))),
                "bootstrap_threshold_max": safe_float(float(np.max(thresholds))),
            }
            add_metric_columns(threshold_base_row, full_oof_metrics, prefix="full_oof_")

            for aggregation, quantile in aggregation_map.items():
                quantile = float(np.clip(float(quantile), 0.0, 1.0))
                bag_threshold = float(np.quantile(thresholds, quantile))
                val_metrics = classification_metrics(y_val, prob_val, threshold=bag_threshold)
                oof_metrics = classification_metrics(y_oof, prob_oof, threshold=bag_threshold)

                threshold_row = {
                    **threshold_base_row,
                    "threshold_aggregation": aggregation,
                    "threshold_quantile": quantile,
                    "threshold": bag_threshold,
                }
                add_metric_columns(threshold_row, oof_metrics, prefix="bag_oof_")
                threshold_rows.append(threshold_row)

                fold_row: Dict[str, object] = {
                    "fold": fold,
                    "bootstrap_objective": objective,
                    "threshold_aggregation": aggregation,
                    "threshold_quantile": quantile,
                    "threshold": bag_threshold,
                    "n_bootstrap": int(n_bootstrap),
                    "n_val": int(len(fold_val)),
                    "n_val_positive": int(np.sum(y_val == 1)),
                    "n_val_negative": int(np.sum(y_val == 0)),
                    "source_threshold": safe_float(pd.to_numeric(fold_val["threshold"], errors="coerce").iloc[0]),
                }
                add_metric_columns(fold_row, val_metrics)
                add_metric_columns(fold_row, oof_metrics, prefix="oof_")
                fold_rows.append(fold_row)

                for _, row in fold_val.iterrows():
                    pred_label = int(float(row["prob"]) >= bag_threshold)
                    pred_rows.append(
                        {
                            "fold": fold,
                            "bootstrap_objective": objective,
                            "threshold_aggregation": aggregation,
                            "threshold_quantile": quantile,
                            "threshold": bag_threshold,
                            "subject_id": normalize_subject_id(row["subject_id"]),
                            "label": float(row["label"]),
                            "prob": float(row["prob"]),
                            "prob_uncalibrated": safe_float(row.get("prob_uncalibrated", math.nan)),
                            "logit_uncalibrated": safe_float(row.get("logit_uncalibrated", math.nan)),
                            "logit_std": safe_float(row.get("logit_std", math.nan)),
                            "pred_label": pred_label,
                            "margin": float(row["prob"]) - bag_threshold,
                            "source_threshold": safe_float(row.get("threshold", math.nan)),
                            "source_pred_label": int(row["pred_label"]) if "pred_label" in row else None,
                        }
                    )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_sample_df = pd.DataFrame(threshold_sample_rows)
    if fold_df.empty or pred_df.empty:
        raise RuntimeError("threshold bagging did not produce any fold results")

    summary_df, summary_json = summarize_threshold_bagging(fold_df, pred_df)
    misclassified_df, feature_df, neighbor_df = generate_threshold_bagging_error_reports(
        config,
        manifest,
        pred_df,
        source_pred_df,
        oof_df,
        top_features=top_features,
        neighbor_k=neighbor_k,
        boundary_margin=boundary_margin,
    )

    fold_df.to_csv(reports_dir / f"{report_prefix}_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(reports_dir / f"{report_prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(reports_dir / f"{report_prefix}_thresholds.csv", index=False, encoding="utf-8-sig")
    threshold_sample_df.to_csv(
        reports_dir / f"{report_prefix}_threshold_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(reports_dir / f"{report_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    misclassified_df.to_csv(
        reports_dir / f"{report_prefix}_misclassified_subjects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_df.to_csv(
        reports_dir / f"{report_prefix}_misclassified_feature_zscores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    neighbor_df.to_csv(
        reports_dir / f"{report_prefix}_misclassified_neighbors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        reports_dir / f"{report_prefix}_summary.json",
        {
            **summary_json,
            "source_prefix": source_prefix,
            "n_bootstrap": int(n_bootstrap),
            "objectives": objectives,
            "aggregations": aggregation_map,
            "top_features": int(top_features),
            "neighbor_k": int(neighbor_k),
            "boundary_margin": float(boundary_margin),
            "outputs": {
                "fold_metrics": f"{report_prefix}_fold_metrics.csv",
                "predictions": f"{report_prefix}_predictions.csv",
                "thresholds": f"{report_prefix}_thresholds.csv",
                "threshold_samples": f"{report_prefix}_threshold_samples.csv",
                "summary": f"{report_prefix}_summary.csv",
                "misclassified_subjects": f"{report_prefix}_misclassified_subjects.csv",
                "misclassified_feature_zscores": f"{report_prefix}_misclassified_feature_zscores.csv",
                "misclassified_neighbors": f"{report_prefix}_misclassified_neighbors.csv",
            },
        },
    )
    return summary_df


def run_biomarker_feature_select(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    logs_dir = outputs["logs"]

    select_cfg = config.get("biomarker_feature_select", {})
    nested_pos_weight = select_cfg.get("positive_class_weight", 1.0)
    nested_use_weighted_sampler = bool(select_cfg.get("use_weighted_sampler", True))
    ensemble_size = max(1, int(select_cfg.get("ensemble_size", 10)))
    ensemble_seed_stride = int(select_cfg.get("ensemble_seed_stride", 10000))
    anchor_current = bool(select_cfg.get("anchor_current_biomarkers", False))
    model_type = str(select_cfg.get("model_type", "linear_residual")).lower()
    if model_type in {"mlp", "biomarker_mlp"}:
        model_builder = build_biomarker_mlp_model
        model_tag = "mlp"
    else:
        model_builder = build_linear_residual_biomarker_model
        model_tag = "residual"
    report_prefix = (
        f"biomarker_stable_select_anchor20_{model_tag}_ensemble{ensemble_size}"
        if anchor_current
        else f"biomarker_stable_select_{model_tag}_ensemble{ensemble_size}"
    )
    ckpt_dir = outputs["checkpoints"] / report_prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_config = make_biomarker_bce_config(
        config,
        positive_class_weight=nested_pos_weight,
        use_weighted_sampler=nested_use_weighted_sampler,
    )
    if "learning_rate" in select_cfg:
        run_config["training"]["learning_rate"] = float(select_cfg["learning_rate"])
    if "weight_decay" in select_cfg:
        run_config["training"]["weight_decay"] = float(select_cfg["weight_decay"])

    inner_splits_requested = int(select_cfg.get("inner_splits", 4))
    calibration_enabled = bool(select_cfg.get("calibration_enabled", True))
    calibration_c = float(select_cfg.get("calibration_c", 1.0))
    final_epoch_aggregation = str(select_cfg.get("final_epoch_aggregation", "median")).lower()
    min_final_epochs = int(select_cfg.get("min_final_epochs", 20))
    max_final_epochs = int(select_cfg.get("max_final_epochs", run_config["training"]["max_epochs"]))
    threshold_strategies = [
        str(x).lower()
        for x in select_cfg.get("strategies", ["accuracy", "balanced_accuracy", "f1", "youden"])
    ]
    primary_strategy = str(select_cfg.get("primary_strategy", threshold_strategies[0])).lower()
    if primary_strategy not in threshold_strategies:
        threshold_strategies = [primary_strategy] + [s for s in threshold_strategies if s != primary_strategy]

    device = choose_device(run_config)
    action_order = get_action_order(run_config)
    subject_df = build_subject_table(manifest, action_order)
    domain_df, domain_feature_cols = load_subject_domain_features(run_config, subject_df)
    feature_matrix = domain_df[domain_feature_cols].to_numpy(dtype=np.float64)
    label_array = subject_df["injury_label"].to_numpy(dtype=np.int64)
    label_tensor = torch.tensor(label_array.astype(np.float32), dtype=torch.float32)

    write_json(
        reports_dir / f"{report_prefix}_features.json",
        {
            "n_candidate_features": len(domain_feature_cols),
            "feature_package": str(run_config["data"]["biomarker_package_zip"]),
            "top_k_values": [int(k) for k in select_cfg.get("top_k_values", [20, 30, 40])],
            "selection_repeats": int(select_cfg.get("selection_repeats", 100)),
            "selection_train_fraction": float(select_cfg.get("selection_train_fraction", 0.8)),
            "corr_threshold": float(select_cfg.get("corr_threshold", 0.9)),
            "anchor_current_biomarkers": anchor_current,
            "model": model_type,
            "ensemble_size": ensemble_size,
            "loss": "bce",
            "positive_class_weight": nested_pos_weight,
            "use_weighted_sampler": nested_use_weighted_sampler,
            "calibration": "platt" if calibration_enabled else "identity",
        },
    )

    batch_size = int(run_config["training"]["batch_size"])
    random_seed = int(run_config["splits"].get("random_seed", 42))

    fold_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    prediction_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    threshold_rows: List[Dict[str, float]] = []
    inner_fold_rows: List[Dict[str, float]] = []
    inner_oof_rows: List[Dict[str, float]] = []
    final_history_rows: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, object]] = []
    k_selection_rows: List[Dict[str, object]] = []

    for fold in sorted(subject_df["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(subject_df["fold"].to_numpy() != fold)[0]
        val_idx = np.where(subject_df["fold"].to_numpy() == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        selected_indices, ranked_df, k_df = choose_stable_feature_subset(
            feature_matrix,
            label_array,
            train_idx,
            domain_feature_cols,
            run_config,
            seed=random_seed + fold * 7919,
        )
        selected_feature_names = [domain_feature_cols[i] for i in selected_indices]
        chosen_k = int(len(selected_indices))

        ranked_out = ranked_df.copy()
        ranked_out.insert(0, "fold", fold)
        selection_rows.extend(ranked_out.to_dict(orient="records"))
        k_out = k_df.copy()
        k_out.insert(0, "fold", fold)
        k_out["selected_features"] = [
            ";".join(selected_feature_names) if bool(row["chosen"]) else ""
            for _, row in k_out.iterrows()
        ]
        k_selection_rows.extend(k_out.to_dict(orient="records"))

        selected_feature_tensor = torch.tensor(
            feature_matrix[:, selected_indices].astype(np.float32),
            dtype=torch.float32,
        )
        outer_train_labels = label_array[train_idx].astype(int)
        class_counts = np.bincount(outer_train_labels, minlength=2)
        inner_splits = int(min(inner_splits_requested, class_counts.min()))
        if inner_splits < 2:
            raise RuntimeError(f"outer fold {fold} has too few minority samples for nested OOF thresholding")

        oof_logits = np.full(train_idx.size, np.nan, dtype=np.float64)
        inner_best_epochs: List[int] = []
        splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed + 3000 + fold)

        for inner_id, (inner_train_rel, inner_val_rel) in enumerate(splitter.split(np.zeros(train_idx.size), outer_train_labels)):
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]
            train_ds, val_ds, _ = normalize_subject_feature_fold_tensors(
                selected_feature_tensor,
                label_tensor,
                inner_train_idx,
                inner_val_idx,
            )
            criterion, loss_metadata = build_binary_classification_loss(
                run_config,
                label_array[inner_train_idx].astype(int),
                device,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(run_config["training"]["num_workers"]),
            )
            inner_logits_by_seed: List[np.ndarray] = []
            inner_losses_by_seed: List[float] = []
            inner_seed_epochs: List[int] = []
            inner_y = np.asarray([], dtype=np.float32)
            for ensemble_id in range(ensemble_size):
                inner_seed = random_seed + fold * 10007 + inner_id * 1009 + ensemble_id * ensemble_seed_stride
                inner_model, inner_info = train_biomarker_mlp_with_early_stopping(
                    run_config,
                    train_ds,
                    val_ds,
                    label_array[inner_train_idx].astype(int),
                    input_dim=chosen_k,
                    device=device,
                    seed=inner_seed,
                    model_builder=model_builder,
                )
                inner_loss, inner_y, _, inner_logits_one = infer_sequence_with_logits(
                    inner_model,
                    val_loader,
                    device,
                    criterion=criterion,
                )
                inner_logits_by_seed.append(inner_logits_one.astype(np.float64))
                inner_losses_by_seed.append(float(inner_loss))
                inner_seed_epochs.append(int(inner_info.get("best_epoch", 0)))

            inner_logits_stack = np.stack(inner_logits_by_seed, axis=0)
            inner_logits = inner_logits_stack.mean(axis=0)
            inner_logit_std = inner_logits_stack.std(axis=0)
            inner_prob = sigmoid_numpy(inner_logits)
            oof_logits[inner_val_rel] = inner_logits.astype(np.float64)
            inner_best_epochs.extend(inner_seed_epochs)
            inner_metrics = classification_metrics(inner_y, inner_prob, threshold=0.5)
            inner_fold_row = {
                "outer_fold": fold,
                "inner_fold": int(inner_id),
                "ensemble_size": int(ensemble_size),
                "n_train": int(inner_train_idx.size),
                "n_val": int(inner_val_idx.size),
                "chosen_k": int(chosen_k),
                "best_epoch": int(round(float(np.median(inner_seed_epochs)))),
                "best_epoch_mean": float(np.mean(inner_seed_epochs)),
                "best_epoch_min": int(np.min(inner_seed_epochs)),
                "best_epoch_max": int(np.max(inner_seed_epochs)),
                "val_loss": float(np.mean(inner_losses_by_seed)),
                "loss_name": str(loss_metadata.get("loss_name", "unknown")),
                "pos_weight": float(loss_metadata.get("loss_pos_weight", math.nan)),
                "threshold": 0.5,
            }
            inner_fold_row.update(inner_metrics)
            inner_fold_rows.append(inner_fold_row)

            inner_subjects = subject_df.iloc[inner_val_idx].reset_index(drop=True)
            for i, row in inner_subjects.iterrows():
                inner_oof_rows.append(
                    {
                        "outer_fold": fold,
                        "inner_fold": int(inner_id),
                        "subject_id": row["subject_id"],
                        "label": float(row["injury_label"]),
                        "chosen_k": int(chosen_k),
                        "oof_logit_uncalibrated": float(inner_logits[i]),
                        "oof_logit_std": float(inner_logit_std[i]),
                        "oof_prob_uncalibrated": float(inner_prob[i]),
                    }
                )

        if not np.isfinite(oof_logits).all():
            raise RuntimeError(f"outer fold {fold} did not produce complete OOF logits")

        calibrator = fit_platt_calibrator(
            oof_logits,
            outer_train_labels,
            c_value=calibration_c,
            enabled=calibration_enabled,
        )
        oof_prob_uncalibrated = sigmoid_numpy(oof_logits)
        oof_prob_calibrated = apply_platt_calibrator(oof_logits, calibrator)
        oof_metrics_uncalibrated = classification_metrics(outer_train_labels, oof_prob_uncalibrated, threshold=0.5)
        oof_metrics_calibrated = classification_metrics(outer_train_labels, oof_prob_calibrated, threshold=0.5)

        if final_epoch_aggregation == "mean":
            final_epochs = int(round(float(np.mean(inner_best_epochs))))
        elif final_epoch_aggregation == "max":
            final_epochs = int(max(inner_best_epochs))
        else:
            final_epochs = int(round(float(np.median(inner_best_epochs))))
        final_epochs = int(np.clip(final_epochs, min_final_epochs, max_final_epochs))

        train_ds, val_ds, norm_state = normalize_subject_feature_fold_tensors(
            selected_feature_tensor,
            label_tensor,
            train_idx,
            val_idx,
        )
        final_criterion, final_loss_metadata = build_binary_classification_loss(run_config, outer_train_labels, device)
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(run_config["training"]["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(run_config["training"]["num_workers"]),
        )
        train_logits_by_seed: List[np.ndarray] = []
        val_logits_by_seed: List[np.ndarray] = []
        val_losses_by_seed: List[float] = []
        final_model_states: List[Dict[str, torch.Tensor]] = []
        train_y_final = np.asarray([], dtype=np.float32)
        val_y = np.asarray([], dtype=np.float32)
        for ensemble_id in range(ensemble_size):
            final_seed = random_seed + fold * 997 + ensemble_id * ensemble_seed_stride
            final_model, final_info = train_biomarker_mlp_fixed_epochs(
                run_config,
                train_ds,
                outer_train_labels,
                input_dim=chosen_k,
                device=device,
                seed=final_seed,
                epochs=final_epochs,
                model_builder=model_builder,
            )
            _, train_y_final, _, train_logits_one = infer_sequence_with_logits(
                final_model,
                train_eval_loader,
                device,
                criterion=final_criterion,
            )
            val_loss_one, val_y, _, val_logits_one = infer_sequence_with_logits(
                final_model,
                val_loader,
                device,
                criterion=final_criterion,
            )
            train_logits_by_seed.append(train_logits_one.astype(np.float64))
            val_logits_by_seed.append(val_logits_one.astype(np.float64))
            val_losses_by_seed.append(float(val_loss_one))
            final_model_states.append(
                {key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()}
            )
            for row in final_info.get("history", []):
                final_history_rows.append({"fold": fold, "ensemble_id": ensemble_id, "seed": final_seed, **row})

        train_logits_stack = np.stack(train_logits_by_seed, axis=0)
        val_logits_stack = np.stack(val_logits_by_seed, axis=0)
        train_logits_uncalibrated = train_logits_stack.mean(axis=0)
        val_logits_uncalibrated = val_logits_stack.mean(axis=0)
        val_logit_std = val_logits_stack.std(axis=0)
        train_prob_uncalibrated = sigmoid_numpy(train_logits_uncalibrated)
        val_prob_uncalibrated = sigmoid_numpy(val_logits_uncalibrated)
        val_loss = float(np.mean(val_losses_by_seed))
        train_prob_calibrated = apply_platt_calibrator(train_logits_uncalibrated, calibrator)
        val_prob_calibrated = apply_platt_calibrator(val_logits_uncalibrated, calibrator)
        val_uncalibrated_metrics = classification_metrics(val_y, val_prob_uncalibrated, threshold=0.5)

        strategy_results = {}
        for strategy in threshold_strategies:
            tuned_threshold, train_score, train_metrics_tuned = tune_threshold(
                outer_train_labels,
                oof_prob_calibrated,
                strategy=strategy,
                config=run_config,
            )
            val_metrics = classification_metrics(val_y, val_prob_calibrated, threshold=tuned_threshold)
            strategy_results[strategy] = {
                "threshold": tuned_threshold,
                "train_score": train_score,
                "train_metrics": train_metrics_tuned,
                "val_metrics": val_metrics,
            }

        checkpoint = {
            "fold": fold,
            "model_state_dicts": final_model_states,
            "model_kwargs": {"input_dim": chosen_k},
            "feature_names": selected_feature_names,
            "feature_indices": [int(i) for i in selected_indices],
            "loss_metadata": final_loss_metadata,
            "normalization": {k: v.cpu() for k, v in norm_state.items()},
            "calibrator": calibrator,
            "thresholds": {
                strategy: float(result["threshold"])
                for strategy, result in strategy_results.items()
            },
            "config": run_config,
            "final_epochs": final_epochs,
            "ensemble_size": ensemble_size,
            "chosen_k": chosen_k,
        }
        torch.save(checkpoint, ckpt_dir / f"fold_{fold}_final.pt")

        val_subjects = subject_df.iloc[val_idx].reset_index(drop=True)
        for strategy, result in strategy_results.items():
            threshold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "chosen_k": int(chosen_k),
                    "threshold": float(result["threshold"]),
                    "oof_objective": float(result["train_score"]),
                    "oof_f1": float(result["train_metrics"]["f1"]),
                    "oof_youden": float(youden_index(result["train_metrics"])),
                    "oof_accuracy": float(result["train_metrics"]["accuracy"]),
                    "val_f1": float(result["val_metrics"]["f1"]),
                    "val_youden": float(youden_index(result["val_metrics"])),
                    "val_accuracy": float(result["val_metrics"]["accuracy"]),
                }
            )

            fold_row = {
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "ensemble_size": int(ensemble_size),
                "inner_splits": int(inner_splits),
                "chosen_k": int(chosen_k),
                "selected_feature_count": int(len(selected_feature_names)),
                "inner_best_epoch_mean": float(np.mean(inner_best_epochs)),
                "inner_best_epoch_median": float(np.median(inner_best_epochs)),
                "final_epochs": int(final_epochs),
                "device": str(device),
                "pos_weight": float(final_loss_metadata.get("loss_pos_weight", math.nan)),
                "weighted_sampler": int(nested_use_weighted_sampler),
                "loss_name": str(final_loss_metadata.get("loss_name", "unknown")),
                "loss_gamma": float(final_loss_metadata.get("loss_gamma", math.nan)),
                "loss_neg_weight": float(final_loss_metadata.get("loss_neg_weight", math.nan)),
                "calibration_method": str(calibrator.get("method", "identity")),
                "calibration_coef": float(calibrator.get("coef", math.nan)),
                "calibration_intercept": float(calibrator.get("intercept", math.nan)),
                "threshold_strategy": strategy,
                "threshold": float(result["threshold"]),
                "train_threshold_objective": float(result["train_score"]),
                "oof_auroc_uncalibrated": float(oof_metrics_uncalibrated["auroc"]),
                "oof_auprc_uncalibrated": float(oof_metrics_uncalibrated["auprc"]),
                "oof_auroc_calibrated": float(oof_metrics_calibrated["auroc"]),
                "oof_auprc_calibrated": float(oof_metrics_calibrated["auprc"]),
                "val_auroc_uncalibrated": float(val_uncalibrated_metrics["auroc"]),
                "val_auprc_uncalibrated": float(val_uncalibrated_metrics["auprc"]),
                "val_loss": float(val_loss),
            }
            fold_row.update(result["val_metrics"])
            fold_rows_by_strategy[strategy].append(fold_row)

            for i, row in val_subjects.iterrows():
                prediction_rows_by_strategy[strategy].append(
                    {
                        "fold": fold,
                        "threshold_strategy": strategy,
                        "chosen_k": int(chosen_k),
                        "threshold": float(result["threshold"]),
                        "subject_id": row["subject_id"],
                        "label": float(row["injury_label"]),
                        "prob": float(val_prob_calibrated[i]),
                        "prob_uncalibrated": float(val_prob_uncalibrated[i]),
                        "logit_uncalibrated": float(val_logits_uncalibrated[i]),
                        "logit_std": float(val_logit_std[i]),
                        "pred_label": int(float(val_prob_calibrated[i]) >= float(result["threshold"])),
                    }
                )

        for row in inner_oof_rows:
            if int(row["outer_fold"]) == fold:
                rel = np.where(subject_df.iloc[train_idx]["subject_id"].astype(str).to_numpy() == str(row["subject_id"]))[0]
                if rel.size > 0:
                    row["oof_prob"] = float(oof_prob_calibrated[int(rel[0])])

    pd.DataFrame(selection_rows).to_csv(
        reports_dir / f"{report_prefix}_feature_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(k_selection_rows).to_csv(
        reports_dir / f"{report_prefix}_k_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(inner_fold_rows).to_csv(
        reports_dir / f"{report_prefix}_inner_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(inner_oof_rows).to_csv(
        reports_dir / f"{report_prefix}_inner_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(final_history_rows).to_csv(
        logs_dir / f"{report_prefix}_final_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(reports_dir / f"{report_prefix}_threshold_tuning.csv", index=False, encoding="utf-8-sig")

    primary_fold_df = None
    for strategy, rows in fold_rows_by_strategy.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(prediction_rows_by_strategy[strategy])
        suffix = "" if strategy == primary_strategy else f"_{strategy}"
        fold_df.to_csv(reports_dir / f"{report_prefix}_fold_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(reports_dir / f"{report_prefix}_val_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
        experiment_summary = {
            "device": str(device),
            "n_trials": int(len(manifest)),
            "n_subjects": int(len(subject_df)),
            "candidate_feature_count": int(len(domain_feature_cols)),
            "model": model_type,
            "positive_class_weight": nested_pos_weight,
            "weighted_sampler": bool(nested_use_weighted_sampler),
            "ensemble_size": int(ensemble_size),
            "calibration": "platt" if calibration_enabled else "identity",
            "stable_feature_selection": True,
            "anchor_current_biomarkers": anchor_current,
            "threshold_strategy": strategy,
            "fold_summary": summarize_fold_table(fold_df),
        }
        write_json(reports_dir / f"{report_prefix}_summary{suffix}.json", experiment_summary)
        if strategy == primary_strategy:
            primary_fold_df = fold_df

    if primary_fold_df is None:
        raise RuntimeError("primary stable selected biomarker results were not generated")
    return primary_fold_df


def run_sequence(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    logs_dir = outputs["logs"]
    ckpt_dir = outputs["checkpoints"] / "sequence"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config)
    tabular_df, tabular_feature_cols = load_aligned_tabular_features(config, manifest)
    tabular_tensor = torch.tensor(
        tabular_df[tabular_feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    write_json(
        reports_dir / "sequence_tabular_features.json",
        {
            "n_features": len(tabular_feature_cols),
            "feature_names": tabular_feature_cols,
        },
    )
    dataset = MultiModalTrialDataset(manifest, config)
    cached = materialize_dataset(dataset, tabular=tabular_tensor)

    threshold = float(config["training"].get("decision_threshold", 0.5))
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stop_patience"])
    min_epochs = int(config["training"].get("min_epochs", 1))
    monitor_metric = str(config["training"].get("monitor_metric", "auroc")).lower()
    use_weighted_sampler = bool(config["training"].get("use_weighted_sampler", False))
    random_seed = int(config["splits"].get("random_seed", 42))
    tuning_cfg = config.get("threshold_tuning", {})
    threshold_tuning_enabled = bool(tuning_cfg.get("enabled", False))
    threshold_strategies = [str(x).lower() for x in tuning_cfg.get("strategies", ["f1"])]
    primary_strategy = str(tuning_cfg.get("primary_strategy", threshold_strategies[0])).lower()
    if primary_strategy not in threshold_strategies:
        threshold_strategies = [primary_strategy] + [s for s in threshold_strategies if s != primary_strategy]

    fold_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    prediction_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    threshold_rows: List[Dict[str, float]] = []

    for fold in sorted(manifest["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(manifest["fold"].to_numpy() != fold)[0]
        val_idx = np.where(manifest["fold"].to_numpy() == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        train_ds, val_ds, norm_state = normalize_fold_tensors(cached, train_idx, val_idx)
        train_labels = cached.label[train_idx].numpy().astype(int)
        train_sampler = None
        train_shuffle = True
        if use_weighted_sampler:
            train_sampler = build_weighted_sampler(train_labels, seed=random_seed + fold)
            train_shuffle = False
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=int(config["training"]["num_workers"]),
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )

        model = build_sequence_model(config, tabular_dim=int(cached.tabular.shape[1])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        criterion, loss_metadata = build_binary_classification_loss(config, train_labels, device)

        best_monitor = -math.inf
        best_epoch = 0
        best_metrics: Dict[str, float] = {}
        epochs_without_improvement = 0
        history_rows: List[Dict[str, float]] = []

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_y, val_prob = infer_sequence(model, val_loader, device, criterion=criterion)
            metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            train_eval_loss, train_y, train_prob = infer_sequence(model, train_eval_loader, device, criterion=criterion)
            train_metrics = classification_metrics(train_y, train_prob, threshold=threshold)

            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
            }
            for key, value in train_metrics.items():
                row[f"train_{key}"] = value
            for key, value in metrics.items():
                row[f"val_{key}"] = value
            history_rows.append(row)

            monitor_value = get_monitor_value(metrics, monitor_metric, val_loss)
            improved = np.isfinite(monitor_value) and (monitor_value > best_monitor)
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_metrics = {"val_loss": val_loss, **metrics}
                checkpoint = {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_kwargs": {"tabular_dim": int(cached.tabular.shape[1])},
                    "tabular_feature_names": tabular_feature_cols,
                    "loss_metadata": loss_metadata,
                    "normalization": {k: v.cpu() for k, v in norm_state.items()},
                    "best_metrics": best_metrics,
                    "config": config,
                }
                torch.save(checkpoint, ckpt_dir / f"fold_{fold}_best.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs and epochs_without_improvement >= patience:
                break

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(logs_dir / f"sequence_fold_{fold}_history.csv", index=False, encoding="utf-8-sig")

        checkpoint = torch.load(ckpt_dir / f"fold_{fold}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, train_y_final, train_prob_final = infer_sequence(model, train_eval_loader, device, criterion=criterion)
        _, val_y, val_prob = infer_sequence(model, val_loader, device, criterion=criterion)

        val_manifest = manifest.iloc[val_idx].reset_index(drop=True)

        if threshold_tuning_enabled:
            strategy_results = {}
            for strategy in threshold_strategies:
                tuned_threshold, train_score, train_metrics_tuned = tune_threshold(
                    train_y_final,
                    train_prob_final,
                    strategy=strategy,
                    config=config,
                )
                val_metrics = classification_metrics(val_y, val_prob, threshold=tuned_threshold)
                strategy_results[strategy] = {
                    "threshold": tuned_threshold,
                    "train_score": train_score,
                    "train_metrics": train_metrics_tuned,
                    "val_metrics": val_metrics,
                }
        else:
            default_metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            strategy_results = {
                primary_strategy: {
                    "threshold": threshold,
                    "train_score": math.nan,
                    "train_metrics": classification_metrics(train_y_final, train_prob_final, threshold=threshold),
                    "val_metrics": default_metrics,
                }
            }

        for strategy, result in strategy_results.items():
            threshold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "threshold": float(result["threshold"]),
                    "train_objective": float(result["train_score"]),
                    "train_f1": float(result["train_metrics"]["f1"]),
                    "train_youden": float(youden_index(result["train_metrics"])),
                    "val_f1": float(result["val_metrics"]["f1"]),
                    "val_youden": float(youden_index(result["val_metrics"])),
                }
            )

            fold_row = {
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "best_epoch": int(best_epoch),
                "device": str(device),
                "pos_weight": float(loss_metadata.get("loss_pos_weight", math.nan)),
                "weighted_sampler": int(use_weighted_sampler),
                "loss_name": str(loss_metadata.get("loss_name", "unknown")),
                "loss_gamma": float(loss_metadata.get("loss_gamma", math.nan)),
                "loss_neg_weight": float(loss_metadata.get("loss_neg_weight", math.nan)),
                "threshold_strategy": strategy,
                "threshold": float(result["threshold"]),
                "train_threshold_objective": float(result["train_score"]),
            }
            if "loss_cb_beta" in loss_metadata:
                fold_row["loss_cb_beta"] = float(loss_metadata["loss_cb_beta"])
            fold_row.update(result["val_metrics"])
            fold_row["val_loss"] = float(best_metrics.get("val_loss", math.nan))
            fold_rows_by_strategy[strategy].append(fold_row)

            for i, row in val_manifest.iterrows():
                prediction_rows_by_strategy[strategy].append(
                    {
                        "fold": fold,
                        "threshold_strategy": strategy,
                        "threshold": float(result["threshold"]),
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "action": row["action"],
                        "label": float(row["injury_label"]),
                        "prob": float(val_prob[i]),
                        "pred_label": int(float(val_prob[i]) >= float(result["threshold"])),
                    }
                )

    primary_fold_df = None
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(reports_dir / "sequence_threshold_tuning.csv", index=False, encoding="utf-8-sig")

    for strategy, rows in fold_rows_by_strategy.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(prediction_rows_by_strategy[strategy])
        suffix = "" if strategy == primary_strategy else f"_{strategy}"
        fold_df.to_csv(reports_dir / f"sequence_fold_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(reports_dir / f"sequence_val_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
        experiment_summary = {
            "device": str(device),
            "n_trials": int(len(manifest)),
            "n_subjects": int(manifest["subject_id"].nunique()),
            "tabular_feature_count": int(len(tabular_feature_cols)),
            "threshold_strategy": strategy,
            "fold_summary": summarize_fold_table(fold_df),
        }
        write_json(reports_dir / f"sequence_summary{suffix}.json", experiment_summary)
        if strategy == primary_strategy:
            primary_fold_df = fold_df

    if primary_fold_df is None:
        raise RuntimeError("primary threshold strategy results were not generated")
    return primary_fold_df


def run_subject_paired(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    outputs = ensure_output_dirs(config)
    reports_dir = outputs["reports"]
    logs_dir = outputs["logs"]

    device = choose_device(config)
    action_order = get_action_order(config)
    action_name_to_index = (
        manifest[["action", "action_index"]].drop_duplicates().set_index("action")["action_index"].astype(int).to_dict()
    )

    tabular_df, tabular_feature_cols = load_aligned_tabular_features(config, manifest)
    tabular_tensor = torch.tensor(
        tabular_df[tabular_feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    trial_dataset = MultiModalTrialDataset(manifest, config)
    cached_trial = materialize_dataset(trial_dataset, tabular=tabular_tensor)
    subject_df = build_subject_table(manifest, action_order)
    biomarker_df, biomarker_feature_cols, ssc_biomarker_cols, vdj_biomarker_cols = load_subject_biomarker_features(
        config,
        subject_df,
    )
    biomarker_tensor = torch.tensor(
        biomarker_df[biomarker_feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    model_spec = resolve_subject_paired_model_spec(
        config,
        biomarker_dim=int(biomarker_tensor.shape[-1]),
        ssc_biomarker_dim=len(ssc_biomarker_cols),
        vdj_biomarker_dim=len(vdj_biomarker_cols),
    )
    experiment_tag = str(model_spec["report_tag"])
    report_prefix = f"subject_paired_{experiment_tag}"
    ckpt_dir = outputs["checkpoints"] / report_prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        reports_dir / f"{report_prefix}_tabular_features.json",
        {
            "n_features": len(tabular_feature_cols),
            "feature_names": tabular_feature_cols,
            "action_order": action_order,
        },
    )
    subject_df.to_csv(reports_dir / f"{report_prefix}_subject_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(
        reports_dir / f"{report_prefix}_biomarker_features.json",
        {
            "n_features": len(biomarker_feature_cols),
            "feature_names": biomarker_feature_cols,
            "ssc_feature_names": ssc_biomarker_cols,
            "vdj_feature_names": vdj_biomarker_cols,
            "feature_package": str(config["data"]["biomarker_package_zip"]),
            "model_variant": model_spec["variant"],
            "uses_biomarker_input": bool(model_spec["uses_biomarker_input"]),
        },
    )
    aux_cfg = build_subject_paired_aux_config(config, action_order, biomarker_feature_cols)
    cached_subject = materialize_subject_pairs(
        subject_df,
        cached_trial,
        action_order,
        action_name_to_index,
        biomarker=biomarker_tensor,
    )

    threshold = float(config["training"].get("decision_threshold", 0.5))
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stop_patience"])
    min_epochs = int(config["training"].get("min_epochs", 1))
    monitor_metric = str(config["training"].get("monitor_metric", "auroc")).lower()
    use_weighted_sampler = bool(config["training"].get("use_weighted_sampler", False))
    random_seed = int(config["splits"].get("random_seed", 42))
    tuning_cfg = config.get("threshold_tuning", {})
    threshold_tuning_enabled = bool(tuning_cfg.get("enabled", False))
    threshold_strategies = [str(x).lower() for x in tuning_cfg.get("strategies", ["f1"])]
    primary_strategy = str(tuning_cfg.get("primary_strategy", threshold_strategies[0])).lower()
    if primary_strategy not in threshold_strategies:
        threshold_strategies = [primary_strategy] + [s for s in threshold_strategies if s != primary_strategy]

    fold_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    prediction_rows_by_strategy: Dict[str, List[Dict[str, float]]] = {s: [] for s in threshold_strategies}
    threshold_rows: List[Dict[str, float]] = []

    for fold in sorted(subject_df["fold"].dropna().unique()):
        fold = int(fold)
        train_idx = np.where(subject_df["fold"].to_numpy() != fold)[0]
        val_idx = np.where(subject_df["fold"].to_numpy() == fold)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        train_ds, val_ds, norm_state = normalize_paired_fold_tensors(cached_subject, train_idx, val_idx)
        train_labels = cached_subject.label[train_idx].numpy().astype(int)
        train_sampler = None
        train_shuffle = True
        if use_weighted_sampler:
            train_sampler = build_weighted_sampler(train_labels, seed=random_seed + fold)
            train_shuffle = False
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=int(config["training"]["num_workers"]),
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )

        model = build_subject_paired_model(
            config,
            tabular_dim=int(cached_subject.tabular.shape[-1]),
            biomarker_dim=int(cached_subject.biomarker.shape[-1]),
            ssc_biomarker_dim=len(ssc_biomarker_cols),
            vdj_biomarker_dim=len(vdj_biomarker_cols),
            action_order=action_order,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        criterion, loss_metadata = build_binary_classification_loss(config, train_labels, device)
        loss_metadata = {
            **loss_metadata,
            "loss_rank_weight": float(aux_cfg.get("ranking_weight", 0.0)),
            "loss_rank_margin": float(aux_cfg.get("ranking_margin", 0.0)),
            "loss_hard_negative_weight": float(aux_cfg.get("hard_negative_weight", 0.0)),
            "loss_hard_negative_margin": float(aux_cfg.get("hard_negative_margin", 0.0)),
            "loss_hard_negative_topk": float(aux_cfg.get("hard_negative_topk", 0)),
            "loss_hard_negative_warmup_epochs": float(aux_cfg.get("hard_negative_warmup_epochs", 0)),
            "loss_ssc_negative_weight": float(aux_cfg.get("ssc_negative_weight", 0.0)),
            "loss_ssc_negative_margin": float(aux_cfg.get("ssc_negative_margin", 0.0)),
            "loss_ssc_negative_warmup_epochs": float(aux_cfg.get("ssc_negative_warmup_epochs", 0)),
            "loss_raw_cls_weight": float(aux_cfg.get("raw_cls_weight", 0.0)),
            "loss_biomarker_cls_weight": float(aux_cfg.get("biomarker_cls_weight", 0.0)),
            "loss_biomarker_aux_weight": float(aux_cfg.get("biomarker_aux_weight", 0.0)),
            "loss_biomarker_aux_delta": float(aux_cfg.get("biomarker_aux_delta", 0.0)),
        }

        best_monitor = -math.inf
        best_epoch = 0
        best_metrics: Dict[str, float] = {}
        epochs_without_improvement = 0
        history_rows: List[Dict[str, float]] = []

        for epoch in range(1, max_epochs + 1):
            epoch_aux_cfg = resolve_subject_paired_aux_for_epoch(aux_cfg, epoch)
            train_loss, train_loss_parts = train_one_epoch_subject_paired(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch_aux_cfg,
            )
            val_loss, val_y, val_prob, val_loss_parts = infer_subject_paired(
                model,
                val_loader,
                device,
                criterion=criterion,
                aux_cfg=epoch_aux_cfg,
            )
            metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            train_eval_loss, train_y, train_prob, train_eval_loss_parts = infer_subject_paired(
                model,
                train_eval_loader,
                device,
                criterion=criterion,
                aux_cfg=epoch_aux_cfg,
            )
            train_metrics = classification_metrics(train_y, train_prob, threshold=threshold)

            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "train_active_rank_weight": float(epoch_aux_cfg.get("ranking_weight", 0.0)),
                "train_active_hard_negative_weight": float(epoch_aux_cfg.get("hard_negative_weight", 0.0)),
                "train_active_ssc_negative_weight": float(epoch_aux_cfg.get("ssc_negative_weight", 0.0)),
            }
            for key, value in train_metrics.items():
                row[f"train_{key}"] = value
            for key, value in train_loss_parts.items():
                row[f"train_{key}"] = value
            for key, value in train_eval_loss_parts.items():
                row[f"train_eval_{key}"] = value
            for key, value in metrics.items():
                row[f"val_{key}"] = value
            for key, value in val_loss_parts.items():
                row[f"val_{key}"] = value
            history_rows.append(row)

            monitor_value = get_monitor_value(metrics, monitor_metric, val_loss)
            improved = np.isfinite(monitor_value) and (monitor_value > best_monitor)
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_metrics = {"val_loss": val_loss, **metrics}
                checkpoint = {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_kwargs": {
                        "tabular_dim": int(cached_subject.tabular.shape[-1]),
                        "biomarker_dim": int(model_spec["biomarker_dim"]),
                        "ssc_biomarker_dim": int(model_spec["ssc_biomarker_dim"]),
                        "vdj_biomarker_dim": int(model_spec["vdj_biomarker_dim"]),
                        "action_order": list(action_order),
                        "variant": model_spec["variant"],
                    },
                    "tabular_feature_names": tabular_feature_cols,
                    "biomarker_feature_names": biomarker_feature_cols,
                    "loss_metadata": loss_metadata,
                    "normalization": {k: v.cpu() for k, v in norm_state.items()},
                    "best_metrics": best_metrics,
                    "config": config,
                }
                torch.save(checkpoint, ckpt_dir / f"fold_{fold}_best.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs and epochs_without_improvement >= patience:
                break

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(logs_dir / f"{report_prefix}_fold_{fold}_history.csv", index=False, encoding="utf-8-sig")

        checkpoint = torch.load(ckpt_dir / f"fold_{fold}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch_aux_cfg = resolve_subject_paired_aux_for_epoch(aux_cfg, int(checkpoint.get("epoch", best_epoch)))
        _, train_y_final, train_prob_final, _ = infer_subject_paired(
            model,
            train_eval_loader,
            device,
            criterion=criterion,
            aux_cfg=best_epoch_aux_cfg,
        )
        _, val_y, val_prob, _ = infer_subject_paired(
            model,
            val_loader,
            device,
            criterion=criterion,
            aux_cfg=best_epoch_aux_cfg,
        )
        val_raw_prob, val_biomarker_prob = infer_subject_paired_branch_prob(model, val_loader, device)
        if not bool(model_spec["uses_biomarker_input"]):
            val_biomarker_prob = np.full_like(val_raw_prob, np.nan)
        val_slot_prob = infer_paired_slot_prob(model, val_loader, device)

        val_subjects = subject_df.iloc[val_idx].reset_index(drop=True)

        if threshold_tuning_enabled:
            strategy_results = {}
            for strategy in threshold_strategies:
                tuned_threshold, train_score, train_metrics_tuned = tune_threshold(
                    train_y_final,
                    train_prob_final,
                    strategy=strategy,
                    config=config,
                )
                val_metrics = classification_metrics(val_y, val_prob, threshold=tuned_threshold)
                strategy_results[strategy] = {
                    "threshold": tuned_threshold,
                    "train_score": train_score,
                    "train_metrics": train_metrics_tuned,
                    "val_metrics": val_metrics,
                }
        else:
            default_metrics = classification_metrics(val_y, val_prob, threshold=threshold)
            strategy_results = {
                primary_strategy: {
                    "threshold": threshold,
                    "train_score": math.nan,
                    "train_metrics": classification_metrics(train_y_final, train_prob_final, threshold=threshold),
                    "val_metrics": default_metrics,
                }
            }

        for strategy, result in strategy_results.items():
            threshold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "threshold": float(result["threshold"]),
                    "train_objective": float(result["train_score"]),
                    "train_f1": float(result["train_metrics"]["f1"]),
                    "train_youden": float(youden_index(result["train_metrics"])),
                    "val_f1": float(result["val_metrics"]["f1"]),
                    "val_youden": float(youden_index(result["val_metrics"])),
                }
            )

            fold_row = {
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "best_epoch": int(best_epoch),
                "device": str(device),
                "pos_weight": float(loss_metadata.get("loss_pos_weight", math.nan)),
                "weighted_sampler": int(use_weighted_sampler),
                "loss_name": str(loss_metadata.get("loss_name", "unknown")),
                "loss_gamma": float(loss_metadata.get("loss_gamma", math.nan)),
                "loss_neg_weight": float(loss_metadata.get("loss_neg_weight", math.nan)),
                "loss_rank_weight": float(loss_metadata.get("loss_rank_weight", 0.0)),
                "loss_rank_margin": float(loss_metadata.get("loss_rank_margin", 0.0)),
                "loss_hard_negative_weight": float(loss_metadata.get("loss_hard_negative_weight", 0.0)),
                "loss_hard_negative_margin": float(loss_metadata.get("loss_hard_negative_margin", 0.0)),
                "loss_hard_negative_topk": float(loss_metadata.get("loss_hard_negative_topk", 0.0)),
                "loss_hard_negative_warmup_epochs": float(loss_metadata.get("loss_hard_negative_warmup_epochs", 0.0)),
                "loss_ssc_negative_weight": float(loss_metadata.get("loss_ssc_negative_weight", 0.0)),
                "loss_ssc_negative_margin": float(loss_metadata.get("loss_ssc_negative_margin", 0.0)),
                "loss_ssc_negative_warmup_epochs": float(loss_metadata.get("loss_ssc_negative_warmup_epochs", 0.0)),
                "loss_raw_cls_weight": float(loss_metadata.get("loss_raw_cls_weight", 0.0)),
                "loss_biomarker_cls_weight": float(loss_metadata.get("loss_biomarker_cls_weight", 0.0)),
                "loss_biomarker_aux_weight": float(loss_metadata.get("loss_biomarker_aux_weight", 0.0)),
                "loss_biomarker_aux_delta": float(loss_metadata.get("loss_biomarker_aux_delta", 0.0)),
                "model_variant": str(model_spec["variant"]),
                "uses_biomarker_input": int(bool(model_spec["uses_biomarker_input"])),
                "biomarker_input_dim": int(model_spec["biomarker_dim"]),
                "biomarker_aux_dim": int(model_spec["biomarker_aux_dim"]),
                "threshold_strategy": strategy,
                "threshold": float(result["threshold"]),
                "train_threshold_objective": float(result["train_score"]),
            }
            if "loss_cb_beta" in loss_metadata:
                fold_row["loss_cb_beta"] = float(loss_metadata["loss_cb_beta"])
            fold_row.update(result["val_metrics"])
            fold_row["val_loss"] = float(best_metrics.get("val_loss", math.nan))
            fold_rows_by_strategy[strategy].append(fold_row)

            for i, row in val_subjects.iterrows():
                valid_slot_probs = []
                for slot_idx, action_name in enumerate(action_order):
                    slot = action_slot_key(action_name)
                    has_slot = int(row[f"has_{slot}"]) == 1
                    valid_slot_probs.append(float(val_slot_prob[i, slot_idx]) if has_slot else math.nan)

                finite_slot_probs = np.asarray(valid_slot_probs, dtype=np.float64)
                finite_mask = np.isfinite(finite_slot_probs)
                if finite_mask.any():
                    max_value = float(np.nanmax(finite_slot_probs))
                    winner_idx = np.where(
                        finite_mask & np.isclose(finite_slot_probs, max_value, atol=1e-8, rtol=1e-6)
                    )[0]
                    max_source = "tie" if len(winner_idx) > 1 else str(action_order[int(winner_idx[0])])
                else:
                    max_value = math.nan
                    max_source = "none"

                pred_row = {
                    "fold": fold,
                    "threshold_strategy": strategy,
                    "threshold": float(result["threshold"]),
                    "subject_id": row["subject_id"],
                    "label": float(row["injury_label"]),
                    "prob": float(val_prob[i]),
                    "prob_raw": float(val_raw_prob[i]),
                    "prob_biomarker": float(val_biomarker_prob[i]),
                    "slot_max_prob": max_value,
                    "max_source": max_source,
                    "pred_label": int(float(val_prob[i]) >= float(result["threshold"])),
                }
                for slot_idx, action_name in enumerate(action_order):
                    slot = action_slot_key(action_name)
                    pred_row[f"trial_id_{slot}"] = row[f"trial_id_{slot}"]
                    pred_row[f"has_{slot}"] = int(row[f"has_{slot}"])
                    pred_row[f"prob_{slot}"] = valid_slot_probs[slot_idx]
                prediction_rows_by_strategy[strategy].append(pred_row)

    primary_fold_df = None
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(reports_dir / f"{report_prefix}_threshold_tuning.csv", index=False, encoding="utf-8-sig")

    for strategy, rows in fold_rows_by_strategy.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(prediction_rows_by_strategy[strategy])
        suffix = "" if strategy == primary_strategy else f"_{strategy}"
        fold_df.to_csv(reports_dir / f"{report_prefix}_fold_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(reports_dir / f"{report_prefix}_val_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
        experiment_summary = {
            "device": str(device),
            "n_trials": int(len(manifest)),
            "n_subjects": int(len(subject_df)),
            "tabular_feature_count": int(len(tabular_feature_cols)),
            "biomarker_feature_count": int(len(biomarker_feature_cols)),
            "model_variant": str(model_spec["variant"]),
            "uses_biomarker_input": bool(model_spec["uses_biomarker_input"]),
            "biomarker_input_dim": int(model_spec["biomarker_dim"]),
            "biomarker_aux_dim": int(model_spec["biomarker_aux_dim"]),
            "action_order": action_order,
            "threshold_strategy": strategy,
            "fold_summary": summarize_fold_table(fold_df),
        }
        write_json(reports_dir / f"{report_prefix}_summary{suffix}.json", experiment_summary)
        if strategy == primary_strategy:
            primary_fold_df = fold_df

    if primary_fold_df is None:
        raise RuntimeError("primary threshold strategy results were not generated")
    return primary_fold_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Training entry for deep learning framework.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "baseline",
            "biomarker_mlp",
            "biomarker_mlp_nested",
            "threshold_bagging",
            "biomarker_feature_select",
            "sequence",
            "subject_paired",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--use-diffusion", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    set_seed(int(config["splits"].get("random_seed", 42)))
    manifest = load_manifest(config)
    ensure_output_dirs(config)

    if args.stage in {"baseline", "all"}:
        baseline_df = run_baseline(config, manifest)
        print("baseline fold metrics:")
        print(baseline_df.to_string(index=False))

    if args.stage in {"biomarker_mlp", "all"}:
        biomarker_mlp_df = run_biomarker_mlp(config, manifest)
        print("biomarker mlp fold metrics:")
        print(biomarker_mlp_df.to_string(index=False))

    if args.stage in {"biomarker_mlp_nested", "all"}:
        biomarker_mlp_nested_df = run_biomarker_mlp_nested(config, manifest)
        print("nested biomarker mlp fold metrics:")
        print(biomarker_mlp_nested_df.to_string(index=False))

    if args.stage in {"threshold_bagging", "all"}:
        threshold_bagging_df = run_threshold_bagging(config, manifest)
        print("threshold bagging summary:")
        print(threshold_bagging_df.to_string(index=False))

    if args.stage in {"biomarker_feature_select", "all"}:
        biomarker_feature_select_df = run_biomarker_feature_select(config, manifest)
        print("stable selected biomarker fold metrics:")
        print(biomarker_feature_select_df.to_string(index=False))

    if args.stage in {"sequence", "all"}:
        sequence_df = run_sequence(config, manifest)
        print("sequence fold metrics:")
        print(sequence_df.to_string(index=False))

    if args.stage == "subject_paired":
        paired_df = run_subject_paired(config, manifest)
        print("subject paired fold metrics:")
        print(paired_df.to_string(index=False))

    if args.use_diffusion:
        print("diffusion flag acknowledged; augmentation hook is reserved for the next implementation step.")


if __name__ == "__main__":
    main()
