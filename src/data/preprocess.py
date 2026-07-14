from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, signal

from src.config import Config
from src.data.read_emg import read_emg, read_emg_metadata
from src.data.read_motion import read_motion


def _not_zone(path: Path) -> bool:
    return not path.name.endswith(":Zone.Identifier")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def _pick_columns(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    picked: list[str] = []
    normed = {c: _norm(c) for c in columns}
    for pat in patterns:
        npat = _norm(pat)
        matches = [c for c, n in normed.items() if npat in n and c not in picked]
        if not matches:
            raise KeyError(f"missing EMG channel pattern: {pat}")
        uv = [c for c in matches if "UV" in normed[c]]
        picked.append((uv or matches)[0])
    return picked


def _numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arr = df[columns].apply(pd.to_numeric, errors="coerce")
    arr = arr.interpolate(limit_direction="both").fillna(0.0)
    return arr.to_numpy(dtype=np.float32)


def clip_by_time(df: pd.DataFrame, start: float, end: float, min_samples: int = 16) -> pd.DataFrame:
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        raise ValueError(f"invalid clip window: start={start}, end={end}")
    part = df[(df["time"] >= start) & (df["time"] <= end)]
    if len(part) < min_samples:
        raise ValueError(
            f"clipped window too short: {len(part)} samples < {min_samples} for [{start}, {end}]"
        )
    return part.copy()


def _estimate_fs(times: np.ndarray, fallback: float) -> float:
    dt = np.diff(times[np.isfinite(times)])
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else fallback


def _filt(data: np.ndarray, fs: float, kind: str, cutoff, order: int) -> np.ndarray:
    if len(data) < 8:
        return data
    nyq = fs / 2.0
    if kind == "bandpass":
        low, high = cutoff
        wn = [max(low / nyq, 1e-5), min(high / nyq, 0.99)]
        b, a = signal.butter(order, wn, btype="bandpass")
    else:
        wn = min(float(cutoff) / nyq, 0.99)
        b, a = signal.butter(order, wn, btype="lowpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen:
        return signal.lfilter(b, a, data, axis=0).astype(np.float32)
    return signal.filtfilt(b, a, data, axis=0).astype(np.float32)


def _interp_to_rate(times: np.ndarray, data: np.ndarray, fs: float) -> np.ndarray:
    order = np.argsort(times)
    times, data = times[order], data[order]
    times, uniq = np.unique(times, return_index=True)
    data = data[uniq]
    if len(times) < 2:
        return np.repeat(data[:1], 4, axis=0)
    n = max(4, int(round((times[-1] - times[0]) * fs)) + 1)
    grid = np.linspace(times[0], times[-1], n)
    fn = interpolate.interp1d(times, data, axis=0, bounds_error=False, fill_value="extrapolate")
    return fn(grid).astype(np.float32)


def fix_length(data: np.ndarray, length: int) -> np.ndarray:
    if data.shape[0] == length:
        return data.astype(np.float32)
    if data.shape[0] < 4:
        data = np.repeat(data[:1], 4, axis=0)
    x = np.linspace(0.0, 1.0, data.shape[0])
    y = np.linspace(0.0, 1.0, length)
    try:
        fn = interpolate.CubicSpline(x, data, axis=0)
        return fn(y).astype(np.float32)
    except ValueError:
        fn = interpolate.interp1d(x, data, axis=0, bounds_error=False, fill_value="extrapolate")
        return fn(y).astype(np.float32)


def find_emg_file(root: Path, subj: str, action: str) -> Path:
    files = sorted(p for p in (root / "肌电").glob(f"{subj}{action}*.csv") if _not_zone(p))
    if not files:
        raise FileNotFoundError(f"missing EMG for {subj} {action}")
    return files[0]


def _trial_from_emg(path: Path) -> str | None:
    match = re.match(r"^(\d{2})(SSC|VDJ)(\d+)", path.name)
    return match.group(3) if match else None


def find_motion_file(root: Path, subj: str, action: str, kind: str, trial: str | None = None) -> Path:
    motion = root / "运动力"
    if kind == "IK":
        files = sorted(p for p in motion.glob(f"{subj}-*-{action}*-IK.mot.csv") if _not_zone(p))
    elif kind == "ID":
        files = sorted(p for p in motion.glob(f"{subj}-*-{action}*-ID.sto.csv") if _not_zone(p))
    elif kind in {"mot", "trc"}:
        files = sorted(p for p in motion.glob(f"{subj}{action}*.{kind}.csv") if _not_zone(p))
    else:
        raise ValueError(kind)
    if not files:
        raise FileNotFoundError(f"missing {kind} for {subj} {action}")
    if trial:
        tagged = [p for p in files if f"{action}{trial}" in p.name]
        if tagged:
            return tagged[0]
    return files[0]


def preprocess_emg(path: Path, start: float, end: float, cfg: Config) -> np.ndarray:
    meta = read_emg_metadata(path)
    fs = float(meta.get("fs", cfg.emg_raw_fs))
    df = clip_by_time(read_emg(path), start, end)
    cols = _pick_columns([c for c in df.columns if c != "time"], cfg.emg_channel_patterns)
    data = _numeric_matrix(df, cols)
    data = _filt(data, fs, "bandpass", cfg.emg_bandpass, cfg.filter_order)
    data = np.abs(data)
    data = _filt(data, fs, "lowpass", cfg.emg_envelope_lp, cfg.filter_order)
    data = _interp_to_rate(df["time"].to_numpy(float), data, cfg.target_fs)
    return fix_length(data, cfg.length).T


def _motion_features(df: pd.DataFrame, count: int) -> np.ndarray:
    cols = [c for c in df.columns if c not in {"time", "frame"}][:count]
    data = _numeric_matrix(df, cols) if cols else np.zeros((len(df), 0), dtype=np.float32)
    if data.shape[1] < count:
        pad = np.zeros((len(df), count - data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
    return data[:, :count]


def preprocess_motion(path: Path, start: float, end: float, lowpass: float, cfg: Config) -> np.ndarray:
    df = clip_by_time(read_motion(path), start, end)
    fs = _estimate_fs(df["time"].to_numpy(float), cfg.motion_fs)
    data = _motion_features(df, cfg.motion_feature_count)
    data = _filt(data, fs, "lowpass", lowpass, cfg.filter_order)
    data = _interp_to_rate(df["time"].to_numpy(float), data, cfg.target_fs)
    return fix_length(data, cfg.length).T


def preprocess_action(root: Path, row: pd.Series, action: str, cfg: Config) -> np.ndarray:
    subj = str(row["subj"]).zfill(2)
    start = float(row[f"{action}_start"])
    end = float(row[f"{action}_end"])
    emg_path = find_emg_file(root, subj, action)
    trial = _trial_from_emg(emg_path)
    ik_path = find_motion_file(root, subj, action, "IK", trial)
    id_path = find_motion_file(root, subj, action, "ID", trial)
    emg = preprocess_emg(emg_path, start, end, cfg)
    ik = preprocess_motion(ik_path, start, end, cfg.ik_lowpass, cfg)
    inv_dyn = preprocess_motion(id_path, start, end, cfg.id_lowpass, cfg)
    out = np.concatenate([emg, ik, inv_dyn], axis=0).astype(np.float32)
    expected = len(cfg.emg_channel_patterns) + 2 * cfg.motion_feature_count
    if out.shape != (expected, cfg.length):
        raise ValueError(f"{subj} {action} produced {out.shape}, expected {(expected, cfg.length)}")
    return out


def preprocess_subject(root: Path, row: pd.Series, cfg: Config) -> np.ndarray:
    actions = [preprocess_action(root, row, action, cfg) for action in cfg.actions]
    return np.stack(actions, axis=0).astype(np.float32)
