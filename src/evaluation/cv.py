from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src.config import Config


@dataclass
class FoldSplit:
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _inner_train_val(trainval_idx: np.ndarray, y: np.ndarray, seed: int, cfg: Config):
    val_n = min(cfg.val_subjects, max(1, len(trainval_idx) // 4))
    rel = np.arange(len(trainval_idx))
    yy = y[trainval_idx]
    if len(np.unique(yy)) >= 2 and min(np.bincount(yy.astype(int))) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_n, random_state=seed)
        tr_rel, va_rel = next(splitter.split(rel, yy))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(rel)
        va_rel, tr_rel = perm[:val_n], perm[val_n:]
    return trainval_idx[tr_rel], trainval_idx[va_rel]


def repeated_subject_splits(
    labels: pd.DataFrame,
    cfg: Config,
    folds: int | None = None,
    repeats: int | None = None,
    repeat_offset: int = 0,
):
    y = labels["label"].to_numpy(int)
    groups = labels["subj"].to_numpy(str)
    x = np.arange(len(labels))
    n_folds = folds or cfg.folds
    n_repeats = repeats or cfg.repeats
    for local_repeat in range(n_repeats):
        repeat = int(repeat_offset) + local_repeat
        outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed + repeat)
        for fold, (trainval_idx, test_idx) in enumerate(outer.split(x, y, groups)):
            seed = cfg.seed + repeat * 100 + fold
            train_idx, val_idx = _inner_train_val(trainval_idx, y, seed, cfg)
            yield FoldSplit(
                repeat=repeat,
                fold=fold,
                seed=seed,
                train_idx=np.asarray(train_idx),
                val_idx=np.asarray(val_idx),
                test_idx=np.asarray(test_idx),
            )
