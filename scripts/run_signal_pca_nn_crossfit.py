#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.labels import load_labels
from src.evaluation.cv import repeated_subject_splits
from src.utils.seed import set_seed


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


FEATURE_STAT_SUFFIXES = (
    "peak_t_ms",
    "early_integral",
    "late_integral",
    "abs_peak",
    "sparsity",
    "range",
    "mean",
    "std",
    "min",
    "max",
    "auc",
)


@dataclass(frozen=True)
class Candidate:
    k: int
    threshold_mode: str
    threshold: float
    flip_scores: bool
    inner_auroc: float
    inner_balanced_accuracy: float
    inner_accuracy: float
    inner_pred_pos_rate: float
    oof_scores: np.ndarray


class SplineLogitClassifier(nn.Module):
    def __init__(self, n_features: int, n_knots: int = 7) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(-3.0, 3.0, n_knots))
        self.log_scale = nn.Parameter(torch.zeros(n_features))
        self.linear = nn.Parameter(torch.empty(n_features))
        self.coeff = nn.Parameter(torch.zeros(n_features, n_knots))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.linear, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.nn.functional.softplus(self.log_scale)[:, None] + 0.35
        basis = torch.exp(
            -0.5 * ((x[:, :, None] - self.centers[None, None, :]) / scale[None, :, :]) ** 2
        )
        basis = basis - basis.mean(dim=2, keepdim=True)
        nonlinear = torch.sum(basis * self.coeff[None, :, :], dim=(1, 2))
        linear = torch.sum(x * self.linear[None, :], dim=1)
        return linear + nonlinear + self.bias

    def regularization_loss(self, spline_decay: float) -> torch.Tensor:
        return (
            torch.sum(self.linear * self.linear)
            + spline_decay * torch.sum(self.coeff * self.coeff)
            + 0.01 * torch.sum(self.log_scale * self.log_scale)
        )


class SplineLogitTrainer:
    def __init__(self, args: argparse.Namespace, seed: int) -> None:
        self.args = args
        self.seed = seed
        self.device = torch.device(args.device)
        self.model: SplineLogitClassifier | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SplineLogitTrainer":
        set_seed(self.seed)
        model = SplineLogitClassifier(x.shape[1], self.args.knots).to(self.device)
        xb = torch.from_numpy(x.astype(np.float32)).to(self.device)
        yb = torch.from_numpy(y.astype(np.float32)).to(self.device)
        n_positive = max(float(y.sum()), 1.0)
        n_negative = max(float(len(y) - y.sum()), 1.0)
        pos_weight = torch.tensor(n_negative / n_positive, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=self.args.learning_rate,
            max_iter=self.args.epochs,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss = loss + self.args.weight_decay * model.regularization_loss(
                self.args.spline_decay
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model = copy.deepcopy(model.eval())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        xb = torch.from_numpy(x.astype(np.float32)).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(xb)).cpu().numpy().astype(float)
        return np.clip(probabilities, 1e-6, 1.0 - 1e-6)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for value in (0, 1):
        mask = labels == value
        recalls.append(float(np.mean(predictions[mask] == value)))
    return float(np.mean(recalls))


def signal_base_name(name: str) -> str:
    for suffix in FEATURE_STAT_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)]
    return name


def compress_signal_groups(
    x_train: np.ndarray,
    x_score: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        groups.setdefault(signal_base_name(str(name)), []).append(index)

    train_parts = []
    score_parts = []
    output_names = []
    for base, indices_list in groups.items():
        indices = np.asarray(indices_list, dtype=int)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_group = scaler.fit_transform(imputer.fit_transform(x_train[:, indices]))
        score_group = scaler.transform(imputer.transform(x_score[:, indices]))
        if train_group.shape[1] == 1:
            train_parts.append(train_group)
            score_parts.append(score_group)
        else:
            reducer = PCA(n_components=1, svd_solver="full")
            train_parts.append(reducer.fit_transform(train_group))
            score_parts.append(reducer.transform(score_group))
        output_names.append(f"{base}__pc1")
    return (
        np.concatenate(train_parts, axis=1).astype(np.float32),
        np.concatenate(score_parts, axis=1).astype(np.float32),
        np.asarray(output_names, dtype=str),
    )


def preprocess_features(
    x_train: np.ndarray,
    x_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(x_train)).astype(np.float32)
    score = scaler.transform(imputer.transform(x_score)).astype(np.float32)
    return train, score


def normalized_rank_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(scores) <= 1:
        return np.ones_like(scores)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores), dtype=float)
    return 1.0 - ranks / float(len(scores) - 1)


def univariate_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    f_scores, _ = f_classif(x, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    auc_scores = np.zeros(x.shape[1], dtype=float)
    for index in range(x.shape[1]):
        if np.std(x[:, index]) > 1e-12:
            auc_scores[index] = abs(safe_auroc(y, x[:, index]) - 0.5)
    return 0.5 * normalized_rank_scores(f_scores) + 0.5 * normalized_rank_scores(auc_scores)


def stratified_subsample(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for value in np.unique(y):
        indices = np.flatnonzero(y == value)
        size = min(len(indices), max(1, int(np.ceil(len(indices) * fraction))))
        selected.append(rng.choice(indices, size=size, replace=False))
    output = np.concatenate(selected)
    rng.shuffle(output)
    return output


def stable_feature_scores(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top_n = min(x.shape[1], max(k, int(np.ceil(k * args.stability_top_mult))))
    frequency = np.zeros(x.shape[1], dtype=float)
    rank_sum = np.zeros(x.shape[1], dtype=float)
    score_sum = np.zeros(x.shape[1], dtype=float)
    used = 0
    for _ in range(args.stability_resamples):
        indices = stratified_subsample(y, args.stability_sample_fraction, rng)
        scores = univariate_scores(x[indices], y[indices])
        score_sum += normalized_rank_scores(scores)
        order = np.argsort(-scores, kind="mergesort")[:top_n]
        frequency[order] += 1.0
        if top_n > 1:
            rank_sum[order] += 1.0 - np.arange(top_n) / float(top_n - 1)
        else:
            rank_sum[order] += 1.0
        used += 1
    return frequency / used + 0.35 * rank_sum / used + 0.15 * score_sum / used


def select_features(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    scores = stable_feature_scores(x, y, k, args, seed)
    order = np.argsort(-scores, kind="mergesort")
    selected: list[int] = []
    for index in order:
        if len(selected) >= min(k, x.shape[1]):
            break
        column = x[:, index]
        if any(
            np.isfinite(correlation := np.corrcoef(column, x[:, previous])[0, 1])
            and abs(float(correlation)) > args.max_correlation
            for previous in selected
        ):
            continue
        selected.append(int(index))
    for index in order:
        if len(selected) >= min(k, x.shape[1]):
            break
        if int(index) not in selected:
            selected.append(int(index))
    return np.asarray(selected, dtype=int)


def prepare_feature_pool(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_score_raw: np.ndarray,
    feature_names: np.ndarray,
    pool_k: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coverage = np.isfinite(x_train_raw).mean(axis=0)
    keep = coverage >= args.minimum_coverage
    if not np.any(keep):
        raise RuntimeError("no features meet the train-fold coverage threshold")
    x_train, x_score, names = compress_signal_groups(
        x_train_raw[:, keep],
        x_score_raw[:, keep],
        feature_names[keep],
    )
    z_train, z_score = preprocess_features(x_train, x_score)
    pool = select_features(z_train, y_train, min(pool_k, z_train.shape[1]), args, seed)
    return z_train, z_score, names, pool


def fit_member_scores(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_score_raw: np.ndarray,
    feature_names: np.ndarray,
    k: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    pool_k = max(k, int(np.ceil(k * args.subspace_pool_multiplier)))
    z_train, z_score, names, pool = prepare_feature_pool(
        x_train_raw,
        y_train,
        x_score_raw,
        feature_names,
        pool_k,
        args,
        seed,
    )
    rng = np.random.default_rng(seed)
    member_scores = []
    selected_union: list[int] = []
    for member in range(args.subspaces):
        if member == 0 or len(pool) <= k:
            selected = pool[: min(k, len(pool))]
        else:
            weights = 1.0 / (np.arange(len(pool), dtype=float) + 1.0)
            weights /= weights.sum()
            selected = rng.choice(pool, size=min(k, len(pool)), replace=False, p=weights)
        classifier = SplineLogitTrainer(args, seed + member * 1009)
        classifier.fit(z_train[:, selected], y_train)
        member_scores.append(classifier.predict_proba(z_score[:, selected]))
        selected_union.extend(int(index) for index in selected)
    return np.stack(member_scores, axis=0), [str(names[index]) for index in sorted(set(selected_union))]


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=float))
    if len(values) == 1:
        return np.asarray([values[0] - 1e-9, values[0], values[0] + 1e-9])
    mids = (values[:-1] + values[1:]) / 2.0
    epsilon = max(float(values[-1] - values[0]) * 1e-6, 1e-9)
    return np.r_[values[0] - epsilon, values, mids, values[-1] + epsilon]


def best_threshold(y: np.ndarray, scores: np.ndarray, mode: str) -> float:
    if mode == "prevalence":
        n_positive = int(np.rint(float(y.mean()) * len(y)))
        ordered = np.sort(scores)[::-1]
        if n_positive <= 0:
            return float(ordered[0] + 1e-9)
        if n_positive >= len(ordered):
            return float(ordered[-1] - 1e-9)
        return float(ordered[n_positive - 1])
    best_key = None
    best_value = 0.5
    for threshold in threshold_grid(scores):
        predictions = (scores >= threshold).astype(int)
        accuracy = float(np.mean(predictions == y))
        balanced = balanced_accuracy(y, predictions)
        prevalence_tie = -abs(float(predictions.mean()) - float(y.mean()))
        key = (accuracy, balanced, prevalence_tie)
        if mode == "balanced_accuracy":
            key = (balanced, accuracy, prevalence_tie)
        if best_key is None or key > best_key:
            best_key = key
            best_value = float(threshold)
    return best_value


def candidate_key(candidate: Candidate, objective: str) -> tuple[float, ...]:
    if objective == "auroc":
        return (
            candidate.inner_auroc,
            candidate.inner_balanced_accuracy,
            candidate.inner_accuracy,
            -candidate.inner_pred_pos_rate,
        )
    joint = (
        candidate.inner_auroc
        + candidate.inner_balanced_accuracy
        + candidate.inner_accuracy
    ) / 3.0
    return (
        joint,
        min(candidate.inner_auroc, candidate.inner_accuracy),
        candidate.inner_balanced_accuracy,
        -candidate.inner_pred_pos_rate,
    )


def score_candidate(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
    k: int,
    args: argparse.Namespace,
    seed: int,
) -> Candidate:
    splitter = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=seed)
    member_oof = np.full((len(y), args.subspaces), np.nan, dtype=float)
    for inner_fold, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
        scores, _ = fit_member_scores(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            feature_names,
            k,
            args,
            seed + inner_fold + 1,
        )
        member_oof[val_idx] = scores.T
    if not np.isfinite(member_oof).all():
        raise RuntimeError("inner OOF contains non-finite scores")
    anchor_weight = 1.0 / args.subspaces
    if args.subspaces == 1:
        raw_oof = member_oof[:, 0]
    else:
        raw_oof = anchor_weight * member_oof[:, 0]
        raw_oof += (1.0 - anchor_weight) * member_oof[:, 1:].mean(axis=1)
    flip_scores = safe_auroc(y, raw_oof) < 0.5
    oriented = 1.0 - raw_oof if flip_scores else raw_oof
    best = None
    for threshold_mode in args.threshold_modes:
        threshold = best_threshold(y, oriented, threshold_mode)
        predictions = (oriented >= threshold).astype(int)
        candidate = Candidate(
            k=k,
            threshold_mode=threshold_mode,
            threshold=threshold,
            flip_scores=flip_scores,
            inner_auroc=safe_auroc(y, oriented),
            inner_balanced_accuracy=balanced_accuracy(y, predictions),
            inner_accuracy=float(np.mean(predictions == y)),
            inner_pred_pos_rate=float(predictions.mean()),
            oof_scores=oriented.copy(),
        )
        if best is None or candidate_key(candidate, args.objective) > candidate_key(best, args.objective):
            best = candidate
    if best is None:
        raise RuntimeError("no candidate selected")
    return best


def percentile_scores(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = []
    for value in values:
        less = float(np.sum(reference < value))
        equal = float(np.sum(reference == value))
        output.append((less + 0.5 * equal + 0.5) / (len(reference) + 1.0))
    return np.asarray(output, dtype=float)


def effect_scores(reference: np.ndarray, labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    negative = reference[labels == 0]
    positive = reference[labels == 1]
    midpoint = 0.5 * (float(negative.mean()) + float(positive.mean()))
    pooled = np.sqrt(0.5 * (float(negative.var()) + float(positive.var())) + 1e-8)
    logits = np.clip((values - midpoint) / pooled, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-logits))


def ecdf_evidence_scores(reference: np.ndarray, labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    negative = reference[labels == 0]
    positive = reference[labels == 1]
    output = []
    for value in values:
        positive_evidence = (
            float(np.sum(negative < value)) + 0.5 * float(np.sum(negative == value)) + 1.0
        ) / (len(negative) + 2.0)
        negative_evidence = (
            float(np.sum(positive > value)) + 0.5 * float(np.sum(positive == value)) + 1.0
        ) / (len(positive) + 2.0)
        output.append(positive_evidence / (positive_evidence + negative_evidence))
    return np.asarray(output, dtype=float)


def configure_branch(args: argparse.Namespace) -> None:
    if args.branch == "ranking":
        args.select_k = [4, 6, 8, 12]
        args.objective = "auroc"
        args.subspaces = 1
        args.subspace_pool_multiplier = 3.0
    else:
        args.select_k = [6]
        args.objective = "joint"
        args.subspaces = 15
        args.subspace_pool_multiplier = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--features", type=Path, default=Path("features/domain_landing_features.csv"))
    parser.add_argument("--branch", choices=["ranking", "classification"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--stability-resamples", type=int, default=10)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.85)
    parser.add_argument("--stability-top-mult", type=float, default=3.0)
    parser.add_argument("--max-correlation", type=float, default=0.95)
    parser.add_argument(
        "--threshold-modes",
        type=lambda value: [part.strip() for part in value.split(",") if part.strip()],
        default=["accuracy", "balanced_accuracy", "prevalence"],
    )
    parser.add_argument("--knots", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.7)
    parser.add_argument("--spline-decay", type=float, default=50.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-outer-splits", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configure_branch(args)
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output exists, refusing to overwrite: {out}")

    labels = load_labels(root / "标签.xlsx")
    feature_path = args.features if args.features.is_absolute() else root / args.features
    table = pd.read_csv(feature_path)
    subjects = table["subj"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).to_numpy()
    expected_subjects = labels["subj"].astype(str).str.zfill(2).to_numpy()
    if not np.array_equal(subjects, expected_subjects):
        raise SystemExit("feature table subject order does not match label table")
    feature_names = np.asarray(
        [column for column in table.columns if column not in {"subj", "label"}],
        dtype=str,
    )
    x = table[feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = table["label"].to_numpy(int)
    cfg = Config(root=root, folds=args.folds, repeats=args.repeats)

    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    split_count = 0
    for split in repeated_subject_splits(
        labels,
        cfg,
        args.folds,
        args.repeats,
        repeat_offset=args.repeat_offset,
    ):
        trainval = np.r_[split.train_idx, split.val_idx]
        y_train = y[trainval]
        best = None
        for k in args.select_k:
            candidate = score_candidate(
                x[trainval],
                y_train,
                feature_names,
                k,
                args,
                split.seed,
            )
            if best is None or candidate_key(candidate, args.objective) > candidate_key(best, args.objective):
                best = candidate
        if best is None:
            raise RuntimeError("no outer-fold candidate selected")

        member_scores, selected_names = fit_member_scores(
            x[trainval],
            y_train,
            x[split.test_idx],
            feature_names,
            best.k,
            args,
            split.seed,
        )
        anchor_weight = 1.0 / args.subspaces
        if args.subspaces == 1:
            raw_scores = member_scores[0]
        else:
            raw_scores = anchor_weight * member_scores[0]
            raw_scores += (1.0 - anchor_weight) * member_scores[1:].mean(axis=0)
        if best.flip_scores:
            raw_scores = 1.0 - raw_scores
        percentile = percentile_scores(best.oof_scores, raw_scores)
        effect = effect_scores(best.oof_scores, y_train, raw_scores)
        ecdf = ecdf_evidence_scores(best.oof_scores, y_train, raw_scores)
        for local_index, subject_index in enumerate(split.test_idx):
            rows.append(
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "seed": split.seed,
                    "subject_index": int(subject_index),
                    "subj": subjects[subject_index],
                    "label": int(y[subject_index]),
                    "raw_score": float(raw_scores[local_index]),
                    "percentile_score": float(percentile[local_index]),
                    "effect_score": float(effect[local_index]),
                    "ecdf_evidence_score": float(ecdf[local_index]),
                    "selected_k": best.k,
                    "feature_set": "all",
                    "flip_scores": best.flip_scores,
                    "inner_auroc": best.inner_auroc,
                    "inner_balanced_acc": best.inner_balanced_accuracy,
                    "inner_accuracy": best.inner_accuracy,
                    "threshold_mode": best.threshold_mode,
                    "subspace_anchor_weight": anchor_weight,
                    "selected_features": ";".join(selected_names),
                }
            )
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        split_count += 1
        print(
            f"branch={args.branch} repeat={split.repeat} fold={split.fold} "
            f"k={best.k} inner_auc={best.inner_auroc:.3f}",
            flush=True,
        )
        if args.max_outer_splits > 0 and split_count >= args.max_outer_splits:
            break

    predictions = pd.DataFrame(rows)
    if args.max_outer_splits == 0:
        aggregated = predictions.groupby("subject_index", as_index=False).agg(
            label=("label", "first"),
            raw_score=("raw_score", "mean"),
            effect_score=("effect_score", "mean"),
            ecdf_evidence_score=("ecdf_evidence_score", "mean"),
        )
        print(
            f"raw_mean_auroc={safe_auroc(aggregated['label'].to_numpy(int), aggregated['raw_score'].to_numpy(float)):.4f} "
            f"raw_mean_auprc={average_precision_score(aggregated['label'], aggregated['raw_score']):.4f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
