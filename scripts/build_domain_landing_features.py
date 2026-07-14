#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.labels import load_labels
from src.data.preprocess import (
    _filt,
    _numeric_matrix,
    _pick_columns,
    _trial_from_emg,
    clip_by_time,
    find_emg_file,
    find_motion_file,
)
from src.data.read_emg import read_emg, read_emg_metadata
from src.data.read_motion import read_motion


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def interp_window(df: pd.DataFrame, columns: list[str], start: float, end: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    part = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    if len(part) < 4:
        part = clip_by_time(df, max(df["time"].min(), start), min(df["time"].max(), end), min_samples=2)
    t = part["time"].to_numpy(float)
    data = part[columns].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    data = data[order]
    t, keep = np.unique(t, return_index=True)
    data = data[keep]
    grid = np.linspace(start, end, points)
    if len(t) == 1:
        return grid, np.repeat(data[:1], points, axis=0)

    out = np.empty((points, data.shape[1]), dtype=float)
    for j in range(data.shape[1]):
        out[:, j] = np.interp(grid, t, data[:, j], left=data[0, j], right=data[-1, j])
    return grid, out


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def safe_name(text: str) -> str:
    name = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return name or "feature"


def add_basic_stats(features: dict[str, float], prefix: str, data: np.ndarray, names: list[str], rel_time: np.ndarray) -> None:
    for j, name in enumerate(names):
        v = data[:, j].astype(float)
        features[f"{prefix}_{name}_mean"] = finite(np.mean(v))
        features[f"{prefix}_{name}_std"] = finite(np.std(v))
        features[f"{prefix}_{name}_min"] = finite(np.min(v))
        features[f"{prefix}_{name}_max"] = finite(np.max(v))
        features[f"{prefix}_{name}_range"] = finite(np.max(v) - np.min(v))
        features[f"{prefix}_{name}_auc"] = finite(np.trapezoid(v, rel_time))
        features[f"{prefix}_{name}_abs_peak"] = finite(np.max(np.abs(v)))
        features[f"{prefix}_{name}_peak_t_ms"] = finite(rel_time[int(np.argmax(np.abs(v)))] * 1000.0)


def hoyer_sparsity(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    n = len(v)
    denom = np.sqrt(n) - 1.0
    if n <= 1 or denom <= 0:
        return 0.0
    l2 = np.linalg.norm(v)
    if l2 <= 1e-12:
        return 0.0
    return float((np.sqrt(n) - (np.sum(np.abs(v)) / l2)) / denom)


def find_contact_time(root: Path, row: pd.Series, action: str) -> tuple[float, pd.DataFrame, str]:
    subj = str(row["subj"]).zfill(2)
    emg_path = find_emg_file(root, subj, action)
    trial = _trial_from_emg(emg_path)
    mot_path = find_motion_file(root, subj, action, "mot", trial)
    mot = read_motion(mot_path)
    start = float(row[f"{action}_start"])
    end = float(row[f"{action}_end"])
    vy_cols = [c for c in mot.columns if c.startswith("ground_force") and c.endswith("_vy")]
    if not vy_cols:
        return start, mot, ""

    label_part = mot[(mot["time"] >= start) & (mot["time"] <= end)]
    scan = label_part if len(label_part) else mot
    active_col = max(vy_cols, key=lambda c: float(scan[c].clip(lower=0).max()))
    force = scan[active_col].to_numpy(float)
    times = scan["time"].to_numpy(float)
    above = np.flatnonzero(force > 20.0)
    if len(above):
        contact = float(times[int(above[0])])
    else:
        contact = float(times[int(np.argmax(force))]) if len(times) else start
    return contact, mot, active_col


def extract_emg_features(root: Path, row: pd.Series, action: str, contact: float, cfg: Config, features: dict[str, float]) -> None:
    subj = str(row["subj"]).zfill(2)
    path = find_emg_file(root, subj, action)
    meta = read_emg_metadata(path)
    fs = float(meta.get("fs", cfg.emg_raw_fs))
    df = read_emg(path)
    cols = _pick_columns([c for c in df.columns if c != "time"], cfg.emg_channel_patterns)

    start = contact - 0.10
    end = contact + 0.20
    part = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    if len(part) < 16:
        start = float(row[f"{action}_start"])
        end = float(row[f"{action}_end"])
        part = clip_by_time(df, start, end)

    raw = _numeric_matrix(part, cols)
    env = _filt(raw, fs, "bandpass", cfg.emg_bandpass, cfg.filter_order)
    env = np.abs(env)
    env = _filt(env, fs, "lowpass", cfg.emg_envelope_lp, cfg.filter_order)
    temp = pd.DataFrame(env, columns=cfg.emg_channel_patterns)
    temp.insert(0, "time", part["time"].to_numpy(float))
    grid, emg = interp_window(temp, list(cfg.emg_channel_patterns), float(temp["time"].min()), float(temp["time"].max()), 101)
    rel = grid - contact
    emg = np.maximum(emg, 0.0)
    add_basic_stats(features, f"{action}_emg", emg, list(cfg.emg_channel_patterns), rel)

    aux_cols = [c for c in df.columns if ("Accel" in str(c) or "Rot" in str(c) or "deg" in str(c))]
    if aux_cols:
        aux_part = part[["time", *aux_cols]].copy()
        _, aux = interp_window(aux_part, aux_cols, float(aux_part["time"].min()), float(aux_part["time"].max()), 101)
        add_basic_stats(features, f"{action}_imu", aux, [safe_name(c) for c in aux_cols], rel)

    name_to_idx = {name: i for i, name in enumerate(cfg.emg_channel_patterns)}
    ta = emg[:, name_to_idx["TIB.ANT"]]
    gast = 0.5 * (emg[:, name_to_idx["MED. GASTRO"]] + emg[:, name_to_idx["LAT. GASTRO"]])
    denom = ta + gast
    cci = np.divide(2.0 * np.minimum(ta, gast), denom, out=np.zeros_like(denom), where=denom > 1e-8)
    early = (rel >= 0.0) & (rel <= 0.05)
    if not early.any():
        early = rel <= np.quantile(rel, 0.5)
    features[f"{action}_ankle_early_cci"] = finite(np.mean(cci[early]))
    features[f"{action}_ankle_full_cci"] = finite(np.mean(cci))

    gmax = float(np.max(gast))
    threshold = max(0.20 * gmax, np.percentile(gast, 60))
    active = gast >= threshold
    if active.any():
        denom_t = max(rel.max() - rel.min(), 1e-6)
        features[f"{action}_gastro_peak_t_pct"] = finite(100.0 * (rel[int(np.argmax(gast))] - rel.min()) / denom_t)
        features[f"{action}_gastro_off_t_pct"] = finite(100.0 * (rel[np.flatnonzero(active)[-1]] - rel.min()) / denom_t)
        features[f"{action}_gastro_duration_ms"] = finite(np.sum(active) * (rel[1] - rel[0]) * 1000.0 if len(rel) > 1 else 0.0)
    else:
        features[f"{action}_gastro_peak_t_pct"] = 0.0
        features[f"{action}_gastro_off_t_pct"] = 0.0
        features[f"{action}_gastro_duration_ms"] = 0.0

    norm = emg / np.maximum(np.max(emg, axis=0, keepdims=True), 1e-8)
    try:
        nmf = NMF(n_components=4, init="nndsvda", max_iter=1000, random_state=cfg.seed)
        w = nmf.fit_transform(norm.T)
        h = nmf.components_
        order = np.argsort(np.argmax(h, axis=1))
        for rank, comp in enumerate(order, start=1):
            features[f"{action}_syn{rank}_sparsity"] = hoyer_sparsity(w[:, comp])
            features[f"{action}_syn{rank}_early_integral"] = finite(np.trapezoid(h[comp, early], rel[early]) if early.any() else 0.0)
            features[f"{action}_syn{rank}_peak_t_ms"] = finite(rel[int(np.argmax(h[comp]))] * 1000.0)
    except Exception:
        for rank in range(1, 5):
            features[f"{action}_syn{rank}_sparsity"] = 0.0
            features[f"{action}_syn{rank}_early_integral"] = 0.0
            features[f"{action}_syn{rank}_peak_t_ms"] = 0.0


def extract_motion_features(
    root: Path,
    row: pd.Series,
    action: str,
    contact: float,
    mot: pd.DataFrame,
    active_force_col: str,
    cfg: Config,
    features: dict[str, float],
) -> None:
    subj = str(row["subj"]).zfill(2)
    trial = _trial_from_emg(find_emg_file(root, subj, action))
    start = contact - 0.10
    end = contact + 0.20
    rel_grid = np.linspace(start, end, 101) - contact

    if active_force_col:
        base = active_force_col[: -len("_vy")]
        cols = [f"{base}_vx", f"{base}_vy", f"{base}_vz"]
        cols = [c for c in cols if c in mot.columns]
        _, force = interp_window(mot, cols, start, end, 101)
        col_names = [c.replace("ground_force", "grf") for c in cols]
        add_basic_stats(features, f"{action}_grf", force, col_names, rel_grid)
        vy_idx = cols.index(active_force_col) if active_force_col in cols else 0
        vgrf = force[:, vy_idx]
        peak_idx = int(np.argmax(vgrf))
        features[f"{action}_vgrf_peak"] = finite(vgrf[peak_idx])
        features[f"{action}_vgrf_peak_t_ms"] = finite(rel_grid[peak_idx] * 1000.0)
        post = rel_grid >= 0.0
        if post.sum() >= 2:
            early_v = vgrf[post]
            early_t = rel_grid[post]
            peak_rel = int(np.argmax(early_v))
            features[f"{action}_vgrf_loading_rate"] = finite((np.max(early_v) - early_v[0]) / max(early_t[peak_rel] - early_t[0], 1e-6))
        else:
            features[f"{action}_vgrf_loading_rate"] = 0.0

    ik_path = find_motion_file(root, subj, action, "IK", trial)
    ik = read_motion(ik_path)
    ik_cols = [
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "knee_adduction_r",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
    ]
    ik_cols = [c for c in ik_cols if c in ik.columns]
    _, ik_data = interp_window(ik, ik_cols, start, end, 101)
    add_basic_stats(features, f"{action}_ik", ik_data, ik_cols, rel_grid)
    if "ankle_angle_r" in ik_cols:
        ankle = ik_data[:, ik_cols.index("ankle_angle_r")]
        features[f"{action}_ankle_pf_rom"] = finite(np.max(ankle) - np.min(ankle))
        features[f"{action}_ankle_early_rom"] = finite(np.max(ankle[rel_grid <= 0.05]) - np.min(ankle[rel_grid <= 0.05]))
    if "knee_angle_r" in ik_cols:
        knee = ik_data[:, ik_cols.index("knee_angle_r")]
        features[f"{action}_knee_flexion_rom"] = finite(np.max(knee) - np.min(knee))

    id_path = find_motion_file(root, subj, action, "ID", trial)
    inv_dyn = read_motion(id_path)
    id_cols = [
        "hip_flexion_r_moment",
        "hip_adduction_r_moment",
        "hip_rotation_r_moment",
        "knee_angle_r_moment",
        "knee_adduction_r_moment",
        "ankle_angle_r_moment",
        "subtalar_angle_r_moment",
    ]
    id_cols = [c for c in id_cols if c in inv_dyn.columns]
    _, id_data = interp_window(inv_dyn, id_cols, start, end, 101)
    add_basic_stats(features, f"{action}_id", id_data, id_cols, rel_grid)


def extract_subject_features(root: Path, row: pd.Series, cfg: Config) -> dict[str, float]:
    features: dict[str, float] = {}
    for action in cfg.actions:
        contact, mot, active_force_col = find_contact_time(root, row, action)
        features[f"{action}_contact_time"] = contact
        features[f"{action}_label_duration"] = float(row[f"{action}_end"] - row[f"{action}_start"])
        features[f"{action}_contact_offset_from_label_start"] = contact - float(row[f"{action}_start"])
        extract_emg_features(root, row, action, contact, cfg, features)
        extract_motion_features(root, row, action, contact, mot, active_force_col, cfg, features)

    if all(k in features for k in ["SSC_ankle_pf_rom", "VDJ_ankle_pf_rom"]):
        features["diff_ankle_pf_rom"] = features["SSC_ankle_pf_rom"] - features["VDJ_ankle_pf_rom"]
        features["mean_ankle_pf_rom"] = 0.5 * (features["SSC_ankle_pf_rom"] + features["VDJ_ankle_pf_rom"])
    if all(k in features for k in ["SSC_vgrf_peak_t_ms", "VDJ_vgrf_peak_t_ms"]):
        features["diff_vgrf_peak_t_ms"] = features["SSC_vgrf_peak_t_ms"] - features["VDJ_vgrf_peak_t_ms"]
        features["mean_vgrf_peak_t_ms"] = 0.5 * (features["SSC_vgrf_peak_t_ms"] + features["VDJ_vgrf_peak_t_ms"])
    if all(k in features for k in ["SSC_ankle_early_cci", "VDJ_ankle_early_cci"]):
        features["diff_ankle_early_cci"] = features["SSC_ankle_early_cci"] - features["VDJ_ankle_early_cci"]
        features["mean_ankle_early_cci"] = 0.5 * (features["SSC_ankle_early_cci"] + features["VDJ_ankle_early_cci"])
    return features


def build_feature_table(root: Path, cfg: Config, out_features: Path, overwrite: bool) -> pd.DataFrame:
    if out_features.exists() and not overwrite:
        raise SystemExit(f"output exists, refusing to overwrite: {out_features}")

    labels = load_labels(root / cfg.label_file)
    rows = []
    for i, (_, row) in enumerate(labels.iterrows(), start=1):
        subj = str(row["subj"]).zfill(2)
        print(f"[{i:02d}/{len(labels):02d}] extracting subject {subj}", flush=True)
        feat = extract_subject_features(root, row, cfg)
        feat["subj"] = subj
        feat["label"] = int(row["label"])
        rows.append(feat)

    table = pd.DataFrame(rows).sort_values("subj").reset_index(drop=True)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_features, index=False, encoding="utf-8-sig")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build domain_landing_features.csv from raw 肌电/ and 运动力/ folders.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project/package root containing 标签.xlsx, 肌电/, and 运动力/.")
    parser.add_argument("--out", type=Path, default=Path("reports/domain_v2/domain_landing_features.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cfg = Config(root=root)
    out = args.out if args.out.is_absolute() else root / args.out

    for required in [root / cfg.label_file, root / cfg.emg_dir, root / cfg.motion_dir]:
        if not required.exists():
            raise SystemExit(f"missing required input: {required}")

    table = build_feature_table(root, cfg, out, args.overwrite)
    feature_cols = [c for c in table.columns if c not in {"subj", "label"}]
    print(f"wrote {out}")
    print(f"subjects={len(table)} features={len(feature_cols)}")


if __name__ == "__main__":
    main()
