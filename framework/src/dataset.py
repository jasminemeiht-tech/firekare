from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset

from .data_io import detect_landing_time, load_emg_csv, load_opensim_csv


@dataclass
class TrialTensors:
    emg: np.ndarray
    mechanics: np.ndarray
    action: int
    label: int
    trial_id: str
    subject_id: str


@dataclass
class SubjectPairedTensors:
    emg: np.ndarray
    mechanics: np.ndarray
    tabular: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    label: int
    subject_id: str
    trial_ids: Tuple[str, ...]


def _find_matching_columns(columns: Iterable[str], targets: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    normalized = {str(c).lower().replace(" ", ""): c for c in columns}
    for target in targets:
        key = target.lower().replace(" ", "")
        matched = None
        for col in columns:
            col_key = str(col).lower().replace(" ", "")
            if key in col_key:
                matched = col
                break
        if matched is None and key in normalized:
            matched = normalized[key]
        if matched is not None:
            mapping[target] = str(matched)
    return mapping


def _linear_resample(data: np.ndarray, target_points: int) -> np.ndarray:
    if data.shape[0] == target_points:
        return data
    x_old = np.linspace(0.0, 1.0, num=data.shape[0])
    x_new = np.linspace(0.0, 1.0, num=target_points)
    out = np.zeros((target_points, data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(x_new, x_old, data[:, i])
    return out


def _infer_fs(time_series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(time_series, errors="coerce")
    diffs = numeric.diff().dropna()
    if diffs.empty:
        return None
    mean_dt = diffs.mean()
    if pd.isna(mean_dt) or mean_dt <= 0:
        return None
    return float(round(1.0 / mean_dt))


def _slice_window(df: Optional[pd.DataFrame], t0: float, pre_ms: float, post_ms: float) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return pd.DataFrame()
    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0
    time = pd.to_numeric(df["time"], errors="coerce")
    window = df[(time >= t0 - pre_s) & (time <= t0 + post_s)].copy()
    if len(window) >= 2:
        return window.reset_index(drop=True)
    return pd.DataFrame()


def _resample_dataframe(df: pd.DataFrame, target_points: int) -> pd.DataFrame:
    if df.empty or "time" not in df.columns or len(df) < 2:
        return df
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    new_time = np.linspace(time.min(), time.max(), target_points)
    out = {"time": new_time}
    for col in df.columns:
        if col == "time":
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        out[col] = np.interp(new_time, time, values)
    return pd.DataFrame(out)


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyquist = 0.5 * fs
    return signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    nyquist = 0.5 * fs
    return signal.butter(order, cutoff_hz / nyquist, btype="low")


def _compute_emg_envelope(raw_emg: Optional[pd.DataFrame], config: dict) -> pd.DataFrame:
    if raw_emg is None or raw_emg.empty or "Time,s" not in raw_emg.columns:
        return pd.DataFrame()

    time = pd.to_numeric(raw_emg["Time,s"], errors="coerce")
    signal_cols = [c for c in raw_emg.columns if c not in {"Time,s", "time"}]
    if not signal_cols:
        return pd.DataFrame()

    emg = raw_emg[signal_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rectified = emg.abs().to_numpy()
    fs = _infer_fs(time)
    if fs is None:
        fs = 2000.0

    band = config["signals"].get("emg_bandpass_hz", [20, 450])
    low_hz = config["signals"].get("emg_lowpass_hz", 6)
    try:
        b_bp, a_bp = _butter_bandpass(float(band[0]), float(band[1]), fs)
        bandpassed = signal.filtfilt(b_bp, a_bp, rectified, axis=0)
        b_lp, a_lp = _butter_lowpass(float(low_hz), fs)
        envelope = signal.filtfilt(b_lp, a_lp, bandpassed, axis=0)
    except Exception:
        window_ms = float(config["signals"].get("emg_envelope_window_ms", 50))
        win_samples = max(int(window_ms * fs / 1000.0), 1)
        kernel = np.ones(win_samples, dtype=np.float32) / float(win_samples)
        envelope = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, rectified)

    env_df = pd.DataFrame(envelope, columns=signal_cols)
    env_df.insert(0, "time", time)
    return env_df.dropna(subset=["time"]).reset_index(drop=True)


def _normalize_emg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df
    out = df.copy()
    for col in out.columns:
        if col == "time":
            continue
        values = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        peak = float(values.abs().max())
        out[col] = values / peak if peak > 0 else 0.0
    return out


def _build_grf_feature_dataframe(grf_df: pd.DataFrame) -> pd.DataFrame:
    if grf_df.empty or "time" not in grf_df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({"time": pd.to_numeric(grf_df["time"], errors="coerce")})
    for axis in ("vx", "vy", "vz"):
        cols = [c for c in grf_df.columns if c.endswith(f"_{axis}")]
        if cols:
            out[f"total_grf_{axis}"] = (
                grf_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            )
        else:
            out[f"total_grf_{axis}"] = 0.0
    return out.dropna(subset=["time"]).reset_index(drop=True)


class MultiModalTrialDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, config: dict):
        self.df = manifest_df.reset_index(drop=True)
        self.config = config
        self.grf_threshold = float(config["signals"]["grf_threshold"])
        self.pre_window_ms = float(config["signals"]["pre_window_ms"])
        self.post_window_ms = float(config["signals"]["post_window_ms"])
        self.target_points = int(config["signals"]["target_points"])
        self.emg_targets = config["signals"]["emg_channels"]
        self.ik_targets = config["signals"]["ik_channels"]
        self.id_targets = config["signals"]["id_channels"]
        self.grf_targets = config["signals"]["grf_channels"]
        self.preload = bool(config["training"].get("preload_dataset", True))
        self._cache: Optional[List[TrialTensors]] = None
        if self.preload:
            self._cache = [self._load_trial(self.df.iloc[i]) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            item = self._cache[index]
        else:
            row = self.df.iloc[index]
            item = self._load_trial(row)
        return {
            "emg": torch.tensor(item.emg, dtype=torch.float32),
            "mechanics": torch.tensor(item.mechanics, dtype=torch.float32),
            "action": torch.tensor(item.action, dtype=torch.long),
            "label": torch.tensor(item.label, dtype=torch.float32),
            "trial_id": item.trial_id,
            "subject_id": item.subject_id,
        }

    def _load_trial(self, row: pd.Series) -> TrialTensors:
        grf_df = load_opensim_csv(Path(row["grf_file"]), file_type="grf")
        landing_t = detect_landing_time(grf_df, threshold=self.grf_threshold)
        if landing_t is None:
            return self._empty_trial(row)

        emg_raw = load_emg_csv(Path(row["emg_file"]))
        ik_df = load_opensim_csv(Path(row["ik_file"]), file_type="ik")
        id_df = load_opensim_csv(Path(row["id_file"]), file_type="id")

        emg_env = _normalize_emg(_compute_emg_envelope(emg_raw, self.config))
        emg_window = _resample_dataframe(
            _slice_window(emg_env, landing_t, self.pre_window_ms, self.post_window_ms),
            self.target_points,
        )
        ik_window = _resample_dataframe(
            _slice_window(ik_df, landing_t, self.pre_window_ms, self.post_window_ms),
            self.target_points,
        )
        id_window = _resample_dataframe(
            _slice_window(id_df, landing_t, self.pre_window_ms, self.post_window_ms),
            self.target_points,
        )
        grf_window = _resample_dataframe(
            _slice_window(grf_df, landing_t, self.pre_window_ms, self.post_window_ms),
            self.target_points,
        )
        grf_features = _build_grf_feature_dataframe(grf_window)

        emg = self._extract_block(emg_window, self.emg_targets)
        ik = self._extract_block(ik_window, self.ik_targets)
        id_block = self._extract_block(id_window, self.id_targets)
        grf = self._extract_block(grf_features, self.grf_targets)

        mechanics = np.concatenate([ik, id_block, grf], axis=1)

        return TrialTensors(
            emg=emg.T,
            mechanics=mechanics.T,
            action=int(row["action_index"]),
            label=int(row["injury_label"]),
            trial_id=str(row["trial_id"]),
            subject_id=str(row["subject_id"]),
        )

    def _empty_trial(self, row: pd.Series) -> TrialTensors:
        emg = np.zeros((len(self.emg_targets), self.target_points), dtype=np.float32)
        mechanics_channels = len(self.ik_targets) + len(self.id_targets) + len(self.grf_targets)
        mechanics = np.zeros((mechanics_channels, self.target_points), dtype=np.float32)
        return TrialTensors(
            emg=emg,
            mechanics=mechanics,
            action=int(row["action_index"]),
            label=int(row["injury_label"]),
            trial_id=str(row["trial_id"]),
            subject_id=str(row["subject_id"]),
        )

    def _extract_block(self, df: Optional[pd.DataFrame], targets: Iterable[str]) -> np.ndarray:
        targets = list(targets)
        if df is None or df.empty:
            return np.zeros((self.target_points, len(targets)), dtype=np.float32)

        mapping = _find_matching_columns(df.columns, targets)
        block = np.zeros((len(df), len(targets)), dtype=np.float32)
        for i, target in enumerate(targets):
            if target in mapping:
                block[:, i] = pd.to_numeric(df[mapping[target]], errors="coerce").fillna(0.0).to_numpy()

        if block.shape[0] == 0:
            return np.zeros((self.target_points, len(targets)), dtype=np.float32)
        block = _linear_resample(block, self.target_points)
        return block.astype(np.float32)


class SubjectPairedTensorDataset(Dataset):
    def __init__(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        tabular: torch.Tensor,
        biomarker: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        label: torch.Tensor,
    ):
        self.emg = emg
        self.mechanics = mechanics
        self.tabular = tabular
        self.biomarker = biomarker
        self.action = action
        self.action_mask = action_mask
        self.label = label

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "emg": self.emg[index],
            "mechanics": self.mechanics[index],
            "tabular": self.tabular[index],
            "biomarker": self.biomarker[index],
            "action": self.action[index],
            "action_mask": self.action_mask[index],
            "label": self.label[index],
        }
