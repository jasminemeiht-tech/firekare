from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    root: Path = Path(".")
    label_file: str = "标签.xlsx"
    emg_dir: str = "肌电"
    motion_dir: str = "运动力"
    cache_dir: str = "cache"
    reports_dir: str = "reports"

    seed: int = 20260505
    actions: tuple[str, str] = ("SSC", "VDJ")
    channel_set_name: str = "main_emg_ik_id"
    length: int = 512
    target_fs: float = 200.0
    emg_raw_fs: float = 2000.0
    motion_fs: float = 100.0

    emg_bandpass: tuple[float, float] = (20.0, 450.0)
    emg_envelope_lp: float = 6.0
    ik_lowpass: float = 6.0
    id_lowpass: float = 12.0
    filter_order: int = 4
    motion_feature_count: int = 15
    emg_channel_patterns: tuple[str, ...] = (
        "VLO",
        "VMO",
        "RECTUS FEM",
        "BICEPS FEM",
        "SEMITEND",
        "TIB.ANT",
        "MED. GASTRO",
        "LAT. GASTRO",
    )

    folds: int = 5
    repeats: int = 10
    val_subjects: int = 6

    embedding_dim: int = 64
    tcn_hidden: int = 64
    tcn_levels: int = 4
    dropout: float = 0.30
    channel_dropout: float = 0.10
    label_smoothing: float = 0.05
    encoder_lr: float = 3e-4
    encoder_weight_decay: float = 1e-4
    encoder_batch_size: int = 8
    encoder_warmup_epochs: int = 5
    encoder_max_epochs: int = 200
    encoder_patience: int = 30

    amplitude_jitter: tuple[float, float] = (0.9, 1.1)
    time_warp: float = 0.05
    noise_std_factor: float = 0.01

    pca_dim: int = 24
    ddpm_steps: int = 100
    ddpm_width: int = 128
    ddpm_blocks: int = 3
    ddpm_cfg_dropout: float = 0.15
    ddpm_guidance_scale: float = 1.5
    ddpm_lr: float = 1e-3
    ddpm_weight_decay: float = 1e-4
    ddpm_batch_size: int = 32
    ddpm_train_steps: int = 5000
    ddpm_ema: float = 0.999

    synth_nn_low_pct: float = 5.0
    synth_nn_high_pct: float = 95.0
    synth_disc_auc_max: float = 0.80
    synth_cap_pos_multiplier: int = 2
    enable_a1: bool = True
    enable_a2: bool = True

    head_lr: float = 1e-3
    head_weight_decay: float = 1e-4
    head_batch_size: int = 16
    head_max_epochs: int = 100
    head_patience: int = 20

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _cache_fingerprint(self) -> str:
        keys = [
            "channel_set_name", "length", "target_fs", "emg_raw_fs", "motion_fs",
            "emg_bandpass", "emg_envelope_lp", "ik_lowpass", "id_lowpass",
            "filter_order", "motion_feature_count", "emg_channel_patterns", "actions",
        ]
        snap = {k: asdict(self)[k] for k in keys}
        h = hashlib.sha1(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return h

    def cache_name(self) -> str:
        return f"{self.channel_set_name}_L{self.length}_fs{int(self.target_fs)}_{self._cache_fingerprint()}.npz"
