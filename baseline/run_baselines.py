from __future__ import annotations

import json
import math
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback only used if xgboost import fails
    XGBClassifier = None


ROOT = Path("/home/ubuntu/xiangmu/xiaofang/chap_2")
FRAMEWORK_ROOT = ROOT / "deeplearning/framework"
BASELINE_ROOT = ROOT / "deeplearning/baseline"
REPORT_ROOT = FRAMEWORK_ROOT / "outputs/reports"
MANIFEST_PATH = FRAMEWORK_ROOT / "outputs/manifests/trials_manifest.csv"
BIOMARKER_ZIP = ROOT / "l2_ligit/biomarker_plus4_logreg_package.zip"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
Q_THRESHOLD = 0.65

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


MODEL_SPECS = [
    {
        "key": "linear_svm",
        "display": "Linear SVM",
        "feature_set": "biomarker20",
    },
    {
        "key": "rbf_svm",
        "display": "RBF-SVM",
        "feature_set": "biomarker20",
    },
    {
        "key": "random_forest",
        "display": "Random Forest",
        "feature_set": "biomarker20",
    },
    {
        "key": "xgboost",
        "display": "XGBoost",
        "feature_set": "biomarker20",
    },
    {
        "key": "mlp_full_features",
        "display": "MLP-full features",
        "feature_set": "full658",
    },
]


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        return str(int(float(text)))
    except ValueError:
        return text.lstrip("0") or "0"


def safe_float(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return float(value)


def load_subject_table() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    subject_df = (
        manifest.groupby("subject_id", sort=True)
        .agg(label=("injury_label", "first"), fold=("fold", "first"), n_trials=("trial_id", "count"))
        .reset_index()
    )
    subject_df["subject_id"] = subject_df["subject_id"].map(normalize_subject_id)
    subject_df["label"] = subject_df["label"].astype(int)
    subject_df["fold"] = subject_df["fold"].astype(int)
    return subject_df


def load_domain_features() -> Tuple[pd.DataFrame, List[str]]:
    with zipfile.ZipFile(BIOMARKER_ZIP) as zf:
        matches = [name for name in zf.namelist() if name.endswith("domain_landing_features.csv")]
        if not matches:
            raise FileNotFoundError(f"domain_landing_features.csv not found inside {BIOMARKER_ZIP}")
        with zf.open(matches[0]) as handle:
            df = pd.read_csv(handle)

    if "subj" not in df.columns or "label" not in df.columns:
        raise ValueError("domain feature table must contain subj and label")
    df = df.copy()
    df["subject_id"] = df["subj"].map(normalize_subject_id)
    feature_cols = [c for c in df.columns if c not in {"subj", "subject_id", "label"}]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return df[["subject_id", "label", *feature_cols]], feature_cols


def align_features(subject_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    domain_df, full_feature_cols = load_domain_features()
    aligned = subject_df[["subject_id", "label", "fold"]].merge(
        domain_df,
        on="subject_id",
        how="left",
        suffixes=("", "_domain"),
        sort=False,
    )
    if aligned[full_feature_cols].isna().all(axis=1).any():
        missing = aligned.loc[aligned[full_feature_cols].isna().all(axis=1), "subject_id"].tolist()
        raise ValueError(f"missing feature rows for subjects: {missing[:5]}")

    label_mismatch = aligned["label_domain"].notna() & (aligned["label_domain"].astype(int) != aligned["label"].astype(int))
    if label_mismatch.any():
        bad = aligned.loc[label_mismatch, "subject_id"].tolist()
        raise ValueError(f"feature labels disagree with manifest labels for subjects: {bad[:5]}")

    missing_biomarkers = [c for c in BIOMARKER_FEATURES if c not in aligned.columns]
    if missing_biomarkers:
        raise ValueError(f"missing biomarker columns: {missing_biomarkers[:5]}")

    y = aligned["label"].to_numpy(dtype=int)
    folds = aligned["fold"].to_numpy(dtype=int)
    x20 = aligned[BIOMARKER_FEATURES].to_numpy(dtype=np.float64)
    xfull = aligned[full_feature_cols].to_numpy(dtype=np.float64)
    return y, folds, x20, xfull, full_feature_cols


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= float(threshold)).astype(int)
    return classification_metrics_from_predictions(y_true, prob, pred)


def classification_metrics_from_predictions(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
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


def threshold_candidates(prob: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 181)
    prob = np.asarray(prob, dtype=float)
    empirical = np.clip(prob[np.isfinite(prob)], 0.05, 0.95)
    return np.unique(np.concatenate([grid, empirical]))


def tune_threshold_fast(y_true: np.ndarray, prob: np.ndarray, strategy: str = "accuracy") -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    candidates = threshold_candidates(prob)
    pred = prob[None, :] >= candidates[:, None]
    positives = y_true[None, :] == 1
    negatives = ~positives

    tp = np.sum(pred & positives, axis=1).astype(np.float64)
    fp = np.sum(pred & negatives, axis=1).astype(np.float64)
    fn = np.sum((~pred) & positives, axis=1).astype(np.float64)
    tn = np.sum((~pred) & negatives, axis=1).astype(np.float64)

    if strategy == "accuracy":
        scores = (tp + tn) / np.maximum(tp + tn + fp + fn, 1.0)
    elif strategy == "f1":
        scores = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)
    else:
        raise ValueError(f"unsupported threshold strategy: {strategy}")

    finite_scores = np.where(np.isfinite(scores), scores, -math.inf)
    best_score = float(np.max(finite_scores))
    tie_idx = np.where(np.isclose(finite_scores, best_score))[0]
    best_idx = int(tie_idx[np.argmin(np.abs(candidates[tie_idx] - 0.5))])
    return float(candidates[best_idx]), best_score


def bootstrap_q_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    q: float = Q_THRESHOLD,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rows = []
    thresholds = []
    for bootstrap_id in range(int(n_bootstrap)):
        sample_idx = np.concatenate(
            [
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
            ]
        )
        rng.shuffle(sample_idx)
        threshold, score = tune_threshold_fast(y_true[sample_idx], prob[sample_idx], strategy="accuracy")
        thresholds.append(threshold)
        rows.append({"bootstrap_id": bootstrap_id, "threshold": threshold, "objective_score": score})
    return float(np.quantile(np.asarray(thresholds, dtype=float), q)), pd.DataFrame(rows)


def make_model(model_key: str, seed: int, y_train: np.ndarray) -> Pipeline:
    pos = max(int(np.sum(y_train == 1)), 1)
    neg = max(int(np.sum(y_train == 0)), 1)
    if model_key == "linear_svm":
        estimator = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=seed)
    elif model_key == "rbf_svm":
        estimator = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    elif model_key == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_key == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available")
        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=1.0,
            scale_pos_weight=float(neg / pos),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    elif model_key == "mlp_full_features":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=3e-4,
            batch_size=8,
            max_iter=600,
            tol=1e-4,
            n_iter_no_change=60,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model key: {model_key}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def predict_positive_prob(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    decision = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -60, 60)))


def run_one_baseline(spec: Dict[str, str], y: np.ndarray, folds: np.ndarray, x: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = BASELINE_ROOT / spec["key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []

    for fold in sorted(np.unique(folds)):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        y_train = y[train_idx]
        class_counts = np.bincount(y_train, minlength=2)
        n_inner = int(min(4, class_counts.min()))
        if n_inner < 2:
            raise RuntimeError(f"fold {fold} has too few minority samples for inner OOF thresholding")

        oof_prob = np.full(train_idx.size, np.nan, dtype=float)
        splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_SEED + 1000 + int(fold))
        for inner_id, (inner_train_rel, inner_val_rel) in enumerate(splitter.split(np.zeros(train_idx.size), y_train)):
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]
            model = make_model(spec["key"], seed=RANDOM_SEED + int(fold) * 1009 + inner_id, y_train=y[inner_train_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x[inner_train_idx], y[inner_train_idx])
            oof_prob[inner_val_rel] = predict_positive_prob(model, x[inner_val_idx])

        if not np.isfinite(oof_prob).all():
            raise RuntimeError(f"incomplete OOF probabilities for {spec['key']} fold {fold}")

        threshold, boot_df = bootstrap_q_threshold(
            y_train,
            oof_prob,
            q=Q_THRESHOLD,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_SEED + int(fold) * 7919,
        )
        boot_df.insert(0, "fold", int(fold))
        bootstrap_rows.extend(boot_df.to_dict(orient="records"))
        full_oof_threshold, full_oof_score = tune_threshold_fast(y_train, oof_prob, strategy="accuracy")
        oof_metrics = classification_metrics(y_train, oof_prob, threshold)
        threshold_rows.append(
            {
                "fold": int(fold),
                "threshold_strategy": "bootstrap_q65_accuracy",
                "threshold": threshold,
                "full_oof_threshold": full_oof_threshold,
                "full_oof_accuracy_score": full_oof_score,
                "bootstrap_threshold_mean": float(boot_df["threshold"].mean()),
                "bootstrap_threshold_std": float(boot_df["threshold"].std(ddof=1)),
                "bootstrap_threshold_median": float(boot_df["threshold"].quantile(0.50)),
                "bootstrap_threshold_q65": float(boot_df["threshold"].quantile(0.65)),
                "bootstrap_threshold_q75": float(boot_df["threshold"].quantile(0.75)),
                **{f"oof_{k}": v for k, v in oof_metrics.items()},
            }
        )

        final_model = make_model(spec["key"], seed=RANDOM_SEED + int(fold), y_train=y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_model.fit(x[train_idx], y_train)
        val_prob = predict_positive_prob(final_model, x[val_idx])
        val_pred = (val_prob >= threshold).astype(int)
        val_metrics = classification_metrics_from_predictions(y[val_idx], val_prob, val_pred)

        fold_row: Dict[str, object] = {
            "model_key": spec["key"],
            "model": spec["display"],
            "feature_set": spec["feature_set"],
            "fold": int(fold),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "threshold": threshold,
        }
        fold_row.update(val_metrics)
        fold_rows.append(fold_row)

        for idx, prob, pred in zip(val_idx, val_prob, val_pred):
            pred_rows.append(
                {
                    "model_key": spec["key"],
                    "model": spec["display"],
                    "feature_set": spec["feature_set"],
                    "fold": int(fold),
                    "subject_index": int(idx),
                    "label": int(y[idx]),
                    "prob": float(prob),
                    "threshold": threshold,
                    "pred_label": int(pred),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    bootstrap_df.to_csv(out_dir / "threshold_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": spec["display"],
        "model_key": spec["key"],
        "feature_set": spec["feature_set"],
        "n_bootstrap": N_BOOTSTRAP,
        "threshold_strategy": "bootstrap_q65_accuracy",
        "fold_summary": summarize_fold_metrics(fold_df),
        "pooled_metrics": pooled_metrics(pred_df),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return fold_df, pred_df


def summarize_fold_metrics(fold_df: pd.DataFrame) -> Dict[str, Dict[str, float | None]]:
    summary = {}
    for col in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1", "tp", "tn", "fp", "fn"]:
        values = pd.to_numeric(fold_df[col], errors="coerce")
        summary[col] = {
            "mean": safe_float(values.mean()),
            "std": safe_float(values.std(ddof=1)),
            "min": safe_float(values.min()),
            "max": safe_float(values.max()),
        }
    return summary


def pooled_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    return classification_metrics_from_predictions(
        pred_df["label"].to_numpy(dtype=int),
        pred_df["prob"].to_numpy(dtype=float),
        pred_df["pred_label"].to_numpy(dtype=int),
    )


def load_current_model_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = REPORT_ROOT / "biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df[pred_df["threshold_aggregation"].astype(str).str.lower() == "q65"].copy()
    pred_df["model_key"] = "current_biomarker_mlp_q65"
    pred_df["model"] = "Biomarker MLP q65"
    pred_df["feature_set"] = "biomarker20"
    pred_df["pred_label"] = pred_df["pred_label"].astype(int)
    pred_df["label"] = pred_df["label"].astype(int)
    fold_rows = []
    for fold, group in pred_df.groupby("fold", sort=True):
        metrics = classification_metrics_from_predictions(
            group["label"].to_numpy(dtype=int),
            group["prob"].to_numpy(dtype=float),
            group["pred_label"].to_numpy(dtype=int),
        )
        fold_row = {
            "model_key": "current_biomarker_mlp_q65",
            "model": "Biomarker MLP q65",
            "feature_set": "biomarker20",
            "fold": int(fold),
            "n_train": 32,
            "n_val": int(len(group)),
            "threshold": float(group["threshold"].iloc[0]),
        }
        fold_row.update(metrics)
        fold_rows.append(fold_row)

    out_dir = BASELINE_ROOT / "current_biomarker_mlp_q65"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    keep_cols = ["model_key", "model", "feature_set", "fold", "label", "prob", "threshold", "pred_label"]
    pred_df[keep_cols].to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "Biomarker MLP q65",
        "model_key": "current_biomarker_mlp_q65",
        "feature_set": "biomarker20",
        "source": str(pred_path),
        "fold_summary": summarize_fold_metrics(fold_df),
        "pooled_metrics": pooled_metrics(pred_df),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return fold_df, pred_df[keep_cols]


def write_comparison_tables(fold_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    rows = []
    for (model_key, model), group in fold_df.groupby(["model_key", "model"], sort=False):
        pooled = pooled_metrics(pred_df[pred_df["model_key"] == model_key])
        row = {"model_key": model_key, "model": model}
        for metric in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = safe_float(values.mean())
            row[f"{metric}_std"] = safe_float(values.std(ddof=1))
            row[f"{metric}_pooled"] = safe_float(pooled[metric])
        for metric in ["tp", "tn", "fp", "fn"]:
            row[f"{metric}_pooled"] = safe_float(pooled[metric])
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(BASELINE_ROOT / "comparison_metrics_summary.csv", index=False, encoding="utf-8-sig")

    table_rows = []
    for _, row in summary_df.iterrows():
        table_rows.append(
            {
                "Model": row["model"],
                "AUROC": f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}",
                "AUPRC": f"{row['auprc_mean']:.3f} ± {row['auprc_std']:.3f}",
                "Accuracy": f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
                "Sensitivity": f"{row['sensitivity_mean']:.3f} ± {row['sensitivity_std']:.3f}",
                "Specificity": f"{row['specificity_mean']:.3f} ± {row['specificity_std']:.3f}",
                "F1": f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f}",
            }
        )
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(BASELINE_ROOT / "comparison_metrics_table.csv", index=False, encoding="utf-8-sig")
    (BASELINE_ROOT / "comparison_metrics_table.md").write_text(table_df.to_markdown(index=False), encoding="utf-8")

    cm_rows = []
    for _, row in summary_df.iterrows():
        cm_rows.append(
            {
                "Model": row["model"],
                "TP": int(row["tp_pooled"]),
                "TN": int(row["tn_pooled"]),
                "FP": int(row["fp_pooled"]),
                "FN": int(row["fn_pooled"]),
            }
        )
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(BASELINE_ROOT / "comparison_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    (BASELINE_ROOT / "comparison_confusion_matrix.md").write_text(cm_df.to_markdown(index=False), encoding="utf-8")


def plot_curves(pred_df: pd.DataFrame) -> None:
    fig_dir = BASELINE_ROOT / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    ordered = [
        "Linear SVM",
        "RBF-SVM",
        "Random Forest",
        "XGBoost",
        "MLP-full features",
        "Biomarker MLP q65",
    ]
    colors = {
        "Linear SVM": "#1f77b4",
        "RBF-SVM": "#ff7f0e",
        "Random Forest": "#2ca02c",
        "XGBoost": "#9467bd",
        "MLP-full features": "#8c564b",
        "Biomarker MLP q65": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for model in ordered:
        group = pred_df[pred_df["model"] == model]
        if group.empty:
            continue
        y_true = group["label"].to_numpy(dtype=int)
        prob = group["prob"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        lw = 2.8 if model == "Biomarker MLP q65" else 1.8
        ax.plot(fpr, tpr, linewidth=lw, color=colors.get(model), label=f"{model} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"roc_curve_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    positive_rate = float(pred_df.drop_duplicates(["model", "fold", "label", "prob"]).query("model == @ordered[0]")["label"].mean())
    for model in ordered:
        group = pred_df[pred_df["model"] == model]
        if group.empty:
            continue
        y_true = group["label"].to_numpy(dtype=int)
        prob = group["prob"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, prob)
        auprc = average_precision_score(y_true, prob)
        lw = 2.8 if model == "Biomarker MLP q65" else 1.8
        ax.plot(recall, precision, linewidth=lw, color=colors.get(model), label=f"{model} (AUPRC={auprc:.3f})")
    ax.axhline(positive_rate, linestyle="--", color="#777777", linewidth=1.0, label=f"Positive rate={positive_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"pr_curve_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    subject_df = load_subject_table()
    y, folds, x20, xfull, full_feature_cols = align_features(subject_df)

    all_fold = []
    all_pred = []
    for spec in MODEL_SPECS:
        x = x20 if spec["feature_set"] == "biomarker20" else xfull
        fold_df, pred_df = run_one_baseline(spec, y, folds, x)
        all_fold.append(fold_df)
        all_pred.append(pred_df)
        print(f"finished {spec['display']}")

    current_fold_df, current_pred_df = load_current_model_predictions()
    if not current_fold_df.empty:
        all_fold.append(current_fold_df)
        all_pred.append(current_pred_df)
        print("included Biomarker MLP q65 current model")

    fold_df = pd.concat(all_fold, ignore_index=True)
    pred_df = pd.concat(all_pred, ignore_index=True)
    fold_df.to_csv(BASELINE_ROOT / "all_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(BASELINE_ROOT / "all_predictions.csv", index=False, encoding="utf-8-sig")
    write_comparison_tables(fold_df, pred_df)
    plot_curves(pred_df)

    metadata = {
        "n_subjects": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "folds": {str(int(fold)): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds))},
        "biomarker20_feature_count": len(BIOMARKER_FEATURES),
        "full_feature_count": len(full_feature_cols),
        "threshold_strategy": "outer-train inner-OOF bootstrap q65 accuracy",
        "xgboost_available": XGBClassifier is not None,
    }
    (BASELINE_ROOT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\ncomparison table:")
    print(pd.read_csv(BASELINE_ROOT / "comparison_metrics_table.csv").to_string(index=False))


if __name__ == "__main__":
    main()
