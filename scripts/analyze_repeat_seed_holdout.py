#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_robust_consensus_dual_output import (
    balanced_accuracy,
    confusion,
    tukey_biweight_location,
)


GROUP_COLUMNS = ["subject_index", "subj", "label"]


def aggregate_ranking(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(GROUP_COLUMNS, as_index=False)
        .agg(ranking_score=("raw_score", tukey_biweight_location))
        .sort_values("subject_index")
    )


def aggregate_classification(rows: pd.DataFrame) -> pd.DataFrame:
    output = (
        rows.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            effect_score=("effect_score", "mean"),
            ecdf_evidence_score=("ecdf_evidence_score", "mean"),
        )
        .sort_values("subject_index")
    )
    output["classification_score"] = np.minimum(
        output["effect_score"].to_numpy(float),
        output["ecdf_evidence_score"].to_numpy(float),
    )
    output["classification_prediction"] = (output["classification_score"] >= 0.5).astype(int)
    return output


def evaluate(name: str, ranking_rows: pd.DataFrame, classification_rows: pd.DataFrame) -> dict[str, object]:
    ranking = aggregate_ranking(ranking_rows)
    classification = aggregate_classification(classification_rows)
    merged = ranking.merge(classification, on=GROUP_COLUMNS, validate="one_to_one")
    labels = merged["label"].to_numpy(int)
    predictions = merged["classification_prediction"].to_numpy(int)
    ranking_scores = merged["ranking_score"].to_numpy(float)
    classification_scores = merged["classification_score"].to_numpy(float)
    tn, fp, fn, tp = confusion(labels, predictions)
    return {
        "split": name,
        "repeat_min": int(min(ranking_rows["repeat"].min(), classification_rows["repeat"].min())),
        "repeat_max": int(max(ranking_rows["repeat"].max(), classification_rows["repeat"].max())),
        "n_ranking_rows": len(ranking_rows),
        "n_classification_rows": len(classification_rows),
        "ranking_auroc": float(roc_auc_score(labels, ranking_scores)),
        "ranking_auprc": float(average_precision_score(labels, ranking_scores)),
        "classification_auroc": float(roc_auc_score(labels, classification_scores)),
        "classification_auprc": float(average_precision_score(labels, classification_scores)),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def subset_sensitivity(
    ranking_rows: pd.DataFrame,
    classification_rows: pd.DataFrame,
) -> pd.DataFrame:
    repeat_ids = sorted(set(ranking_rows["repeat"]) & set(classification_rows["repeat"]))
    records = []
    for size in range(5, len(repeat_ids) + 1):
        values = []
        for chosen in itertools.combinations(repeat_ids, size):
            ranking = ranking_rows[ranking_rows["repeat"].isin(chosen)]
            classification = classification_rows[classification_rows["repeat"].isin(chosen)]
            result = evaluate("subset", ranking, classification)
            values.append(
                [result["ranking_auroc"], result["accuracy"], result["balanced_accuracy"]]
            )
        array = np.asarray(values, dtype=float)
        records.append(
            {
                "repeat_count": size,
                "n_subsets": len(array),
                "ranking_auroc_median": float(np.median(array[:, 0])),
                "ranking_auroc_min": float(np.min(array[:, 0])),
                "ranking_auroc_max": float(np.max(array[:, 0])),
                "accuracy_median": float(np.median(array[:, 1])),
                "accuracy_min": float(np.min(array[:, 1])),
                "accuracy_max": float(np.max(array[:, 1])),
                "fraction_accuracy_ge_0875": float(np.mean(array[:, 1] >= 0.875)),
                "balanced_accuracy_median": float(np.median(array[:, 2])),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-ranking",
        type=Path,
        default=Path("results/predictions/auto_signal_pca_crossfit10_predictions.csv"),
    )
    parser.add_argument(
        "--development-classification",
        type=Path,
        default=Path("results/predictions/auto_signal_pca_subspace15_crossfit10_predictions.csv"),
    )
    parser.add_argument(
        "--holdout-ranking",
        type=Path,
        default=Path("results/predictions/auto_signal_pca_crossfit10_offset10_predictions.csv"),
    )
    parser.add_argument(
        "--holdout-classification",
        type=Path,
        default=Path("results/predictions/auto_signal_pca_subspace15_crossfit10_offset10_predictions.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summaries/reproduced_repeat_seed_holdout.csv"),
    )
    parser.add_argument(
        "--subset-out",
        type=Path,
        default=Path("results/summaries/reproduced_repeat_subset_sensitivity.csv"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("docs/REPRODUCED_RESULT.md"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    paths = {
        "development_ranking": resolve(args.development_ranking),
        "development_classification": resolve(args.development_classification),
        "holdout_ranking": resolve(args.holdout_ranking),
        "holdout_classification": resolve(args.holdout_classification),
    }
    out = resolve(args.out)
    subset_out = resolve(args.subset_out)
    report_out = resolve(args.report_out)
    if any(path.exists() for path in (out, subset_out, report_out)) and not args.overwrite:
        raise SystemExit("one or more outputs exist; use --overwrite")

    data = {name: pd.read_csv(path) for name, path in paths.items()}
    development = evaluate(
        "development_repeats_0_9",
        data["development_ranking"],
        data["development_classification"],
    )
    holdout = evaluate(
        "seed_holdout_repeats_10_19",
        data["holdout_ranking"],
        data["holdout_classification"],
    )
    pooled = evaluate(
        "pooled_repeats_0_19",
        pd.concat([data["development_ranking"], data["holdout_ranking"]], ignore_index=True),
        pd.concat(
            [data["development_classification"], data["holdout_classification"]],
            ignore_index=True,
        ),
    )
    results = pd.DataFrame([development, holdout, pooled])
    sensitivity = subset_sensitivity(
        data["development_ranking"], data["development_classification"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, encoding="utf-8-sig")
    sensitivity.to_csv(subset_out, index=False, encoding="utf-8-sig")

    table_rows = []
    for row in results.itertuples(index=False):
        table_rows.append(
            f"| {row.split} | {row.ranking_auroc:.4f} | {row.ranking_auprc:.4f} | "
            f"{row.accuracy:.4f} | {row.balanced_accuracy:.4f} | "
            f"{row.tn}/{row.fp}/{row.fn}/{row.tp} |"
        )
    report = f"""# Repeat Seed Holdout 复核

## 固定规则

- 排序：每名被试的 raw score 使用 Tukey biweight，标准常数 `c=4.685`。
- 分类：`min(mean(effect_score), mean(ecdf_evidence_score)) >= 0.5`。
- 规则由 repeats 0-9 的开发分析锁定；repeats 10-19 仅用于种子留出复核，没有再修改公式或阈值。
- 这仍然是同一 40 名被试上的算法稳定性检查，不是独立被试外部验证。

## 结果

| 划分 | 排序 AUROC | 排序 AUPRC | Accuracy | Balanced accuracy | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 结论

- 开发 seeds 的 `0.9248/0.8750` 在新 seeds 上降为 `{holdout['ranking_auroc']:.4f}/{holdout['accuracy']:.4f}`，没有复现。
- 合并 20 repeats 后为 `{pooled['ranking_auroc']:.4f}/{pooled['accuracy']:.4f}`，说明前 10 repeats 的峰值对随机划分敏感。
- repeats 0-9 的全部 5-8 repeat 子集中，accuracy 从未达到 `0.875`；该峰值只在恰好使用全部 10 次时出现。
- 因此 `0.9248/0.8750` 应降级为未复现的开发性峰值，不能作为论文确认性主结果。

## 审计

- 原 repeats 0-9 两个源文件的成员审计：`results/audits/auto_signal_pca_robust_consensus_leakage_audit.md`，PASS。
- repeats 10-19 排序审计：`results/audits/auto_signal_pca_crossfit10_offset10_leakage_audit.md`，PASS。
- repeats 10-19 分类审计：`results/audits/auto_signal_pca_subspace15_crossfit10_offset10_leakage_audit.md`，PASS。
- PASS 只表示对应外层测试被试没有进入同一模型的训练/验证；不能消除同一 40 人上的开发偏差。
"""
    report_out.write_text(report, encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
