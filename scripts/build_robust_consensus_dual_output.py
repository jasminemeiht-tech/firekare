#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def tukey_biweight_location(values: np.ndarray, tuning_constant: float = 4.685) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return median
    normalized = (values - median) / (tuning_constant * mad)
    keep = np.abs(normalized) < 1.0
    if not keep.any():
        return median
    weights = (1.0 - normalized[keep] ** 2) ** 2
    if float(weights.sum()) <= 1e-12:
        return median
    return float(np.sum(weights * values[keep]) / np.sum(weights))


def balanced_accuracy(y_true: np.ndarray, predictions: np.ndarray) -> float:
    specificity = float(np.mean(predictions[y_true == 0] == 0))
    sensitivity = float(np.mean(predictions[y_true == 1] == 1))
    return 0.5 * (specificity + sensitivity)


def confusion(y_true: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    return tn, fp, fn, tp


def stratified_bootstrap(
    labels: np.ndarray,
    metric,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    values = []
    for _ in range(samples):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        values.append(float(metric(indices)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking-predictions",
        type=Path,
        default=Path("reports/auto_signal_pca_crossfit10_predictions.csv"),
    )
    parser.add_argument(
        "--classification-predictions",
        type=Path,
        default=Path("reports/auto_signal_pca_subspace15_crossfit10_predictions.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/auto_signal_pca_robust_consensus_dual_output.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/auto_signal_pca_robust_consensus_dual_output_summary.csv"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/auto_signal_pca_robust_consensus_dual_output.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    ranking_path = resolve(args.ranking_predictions)
    classification_path = resolve(args.classification_predictions)
    out = resolve(args.out)
    summary_out = resolve(args.summary_out)
    report_out = resolve(args.report_out)
    if any(path.exists() for path in (out, summary_out, report_out)) and not args.overwrite:
        raise SystemExit("one or more outputs exist; use --overwrite")

    ranking_rows = pd.read_csv(ranking_path)
    classification_rows = pd.read_csv(classification_path)
    required_ranking = {"subject_index", "subj", "label", "raw_score", "repeat", "fold"}
    required_classification = {
        "subject_index",
        "subj",
        "label",
        "effect_score",
        "ecdf_evidence_score",
        "repeat",
        "fold",
    }
    if missing := required_ranking - set(ranking_rows.columns):
        raise RuntimeError(f"ranking predictions missing columns: {sorted(missing)}")
    if missing := required_classification - set(classification_rows.columns):
        raise RuntimeError(f"classification predictions missing columns: {sorted(missing)}")

    grouping = ["subject_index", "subj", "label"]
    ranking = (
        ranking_rows.groupby(grouping, as_index=False)
        .agg(
            ranking_score=("raw_score", tukey_biweight_location),
            ranking_predictions=("raw_score", "size"),
        )
        .sort_values("subject_index")
    )
    classification = (
        classification_rows.groupby(grouping, as_index=False)
        .agg(
            effect_score=("effect_score", "mean"),
            ecdf_evidence_score=("ecdf_evidence_score", "mean"),
            classification_predictions=("effect_score", "size"),
        )
        .sort_values("subject_index")
    )
    merged = ranking.merge(classification, on=grouping, how="inner", validate="one_to_one")
    if len(merged) != 40:
        raise RuntimeError(f"expected 40 subjects, found {len(merged)}")
    if not (merged["ranking_predictions"].eq(10).all() and merged["classification_predictions"].eq(10).all()):
        raise RuntimeError("every subject must have exactly 10 predictions from each branch")

    merged["classification_score"] = np.minimum(
        merged["effect_score"].to_numpy(float),
        merged["ecdf_evidence_score"].to_numpy(float),
    )
    merged["classification_prediction"] = (merged["classification_score"] >= 0.5).astype(int)
    y_true = merged["label"].to_numpy(int)
    ranking_scores = merged["ranking_score"].to_numpy(float)
    classification_scores = merged["classification_score"].to_numpy(float)
    predictions = merged["classification_prediction"].to_numpy(int)
    tn, fp, fn, tp = confusion(y_true, predictions)

    ranking_auroc = float(roc_auc_score(y_true, ranking_scores))
    ranking_auprc = float(average_precision_score(y_true, ranking_scores))
    classification_auroc = float(roc_auc_score(y_true, classification_scores))
    classification_auprc = float(average_precision_score(y_true, classification_scores))
    accuracy = float(np.mean(predictions == y_true))
    balanced_acc = balanced_accuracy(y_true, predictions)
    sensitivity = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))

    auroc_ci = stratified_bootstrap(
        y_true,
        lambda index: roc_auc_score(y_true[index], ranking_scores[index]),
        args.bootstrap_samples,
        args.seed,
    )
    accuracy_ci = stratified_bootstrap(
        y_true,
        lambda index: np.mean(predictions[index] == y_true[index]),
        args.bootstrap_samples,
        args.seed + 1,
    )
    balanced_ci = stratified_bootstrap(
        y_true,
        lambda index: balanced_accuracy(y_true[index], predictions[index]),
        args.bootstrap_samples,
        args.seed + 2,
    )

    summary = pd.DataFrame(
        [
            {
                "pipeline": "robust_ranking_plus_conservative_calibration_consensus",
                "subjects": len(merged),
                "folds": 5,
                "repeats": 10,
                "ranking_aggregation": "Tukey biweight c=4.685 over 10 raw scores",
                "classification_aggregation": "min(mean effect, mean ECDF evidence)",
                "classification_threshold": 0.5,
                "ranking_auroc": ranking_auroc,
                "ranking_auroc_ci95_low": auroc_ci[0],
                "ranking_auroc_ci95_high": auroc_ci[1],
                "ranking_auprc": ranking_auprc,
                "classification_auroc": classification_auroc,
                "classification_auprc": classification_auprc,
                "accuracy": accuracy,
                "accuracy_ci95_low": accuracy_ci[0],
                "accuracy_ci95_high": accuracy_ci[1],
                "balanced_accuracy": balanced_acc,
                "balanced_accuracy_ci95_low": balanced_ci[0],
                "balanced_accuracy_ci95_high": balanced_ci[1],
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "ranking_source_sha256": sha256(ranking_path),
                "classification_source_sha256": sha256(classification_path),
            }
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    report = f"""# 稳健排序与保守校准共识双输出结果

## 方法

- 评估基础：subject-level 5-fold x 10 repeats；每名被试有 10 个由未见过该被试的外层模型产生的预测。
- 排序输出：对非子空间分支的 10 个 `raw_score` 使用 Tukey biweight（标准常数 `c=4.685`）稳健位置估计。
- 分类输出：15 子空间分支先分别对 10 个 `effect_score` 和 `ecdf_evidence_score` 求均值，再取两者较小值；固定阈值 `0.5`。这等价于只有两种训练内校准都支持阳性时才判为阳性。
- 聚合和共识公式不读取目标被试标签；PCA、缺失处理、稳定选择、方向判断、校准和神经网络训练均发生在对应外层训练数据内。

## 结果

| 输出 | 指标 | 结果 |
|---|---|---:|
| 排序 | AUROC | {ranking_auroc:.4f}（bootstrap 95% CI {auroc_ci[0]:.4f}-{auroc_ci[1]:.4f}） |
| 排序 | AUPRC | {ranking_auprc:.4f} |
| 分类 | Accuracy | {accuracy:.4f}（bootstrap 95% CI {accuracy_ci[0]:.4f}-{accuracy_ci[1]:.4f}） |
| 分类 | Balanced accuracy | {balanced_acc:.4f}（bootstrap 95% CI {balanced_ci[0]:.4f}-{balanced_ci[1]:.4f}） |
| 分类 | Sensitivity / Specificity | {sensitivity:.4f} / {specificity:.4f} |
| 分类 | TN / FP / FN / TP | {tn} / {fp} / {fn} / {tp} |

分类共识分数自身的 AUROC/AUPRC 为 {classification_auroc:.4f}/{classification_auprc:.4f}；论文中不得将它与排序分支的 AUROC 混写成同一个单输出模型指标。

## 解释边界

- 与上一双输出候选 `0.9216 / 0.8500` 相比，本方案在当前 40 人上达到 `0.9248 / 0.8750`。
- 独立审计 `reports/auto_signal_pca_robust_consensus_leakage_audit.md` 已核验两个源文件：各 400 行、每人 10 条、外层成员关系违规 0，结论 PASS。
- 这是同一 40 人上继续比较聚合与校准规则后得到的开发性结果。它没有直接把外层测试被试用于对应模型训练，但存在实验级 meta-selection bias，必须锁定规则后由新增被试或外部队列确认。
- 后续使用全新 repeats 10-19 的固定规则复核未能复现该峰值；详见 `reports/auto_signal_pca_repeat_seed_holdout.md`。因此本文件只保留开发集结果，不作为稳定主结果。
"""
    report_out.write_text(report, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"wrote {out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
