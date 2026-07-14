#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.labels import load_labels
from src.evaluation.cv import repeated_subject_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def audit_file(
    path: Path,
    labels: pd.DataFrame,
    cfg: Config,
    folds: int,
    repeats: int,
    repeat_offset: int,
) -> dict[str, object]:
    rows = pd.read_csv(path)
    expected_columns = {"repeat", "fold", "subject_index", "subj", "label"}
    missing = expected_columns - set(rows.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")

    split_map = {
        (split.repeat, split.fold): split
        for split in repeated_subject_splits(labels, cfg, folds, repeats, repeat_offset=repeat_offset)
    }
    violations = []
    subjects = labels["subj"].astype(str).str.zfill(2).to_numpy()
    y = labels["label"].to_numpy(int)
    for row in rows.itertuples(index=False):
        key = (int(row.repeat), int(row.fold))
        if key not in split_map:
            violations.append(f"unknown split {key}")
            continue
        split = split_map[key]
        subject_idx = int(row.subject_index)
        if subject_idx not in set(split.test_idx.tolist()):
            violations.append(f"subject {subject_idx} not in test for split {key}")
        if subject_idx in set(np.r_[split.train_idx, split.val_idx].tolist()):
            violations.append(f"subject {subject_idx} appears in train/val for split {key}")
        if str(row.subj).zfill(2) != subjects[subject_idx]:
            violations.append(f"subject id mismatch at index {subject_idx}")
        if int(row.label) != int(y[subject_idx]):
            violations.append(f"label mismatch at index {subject_idx}")

    duplicates = int(rows.duplicated(["repeat", "subject_index"]).sum())
    counts = rows.groupby("subject_index").size()
    complete_subjects = int((counts == repeats).sum())
    expected_rows = len(labels) * repeats
    return {
        "path": str(path),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_subjects": int(rows["subject_index"].nunique()),
        "subjects_with_expected_predictions": complete_subjects,
        "duplicate_subject_repeat_rows": duplicates,
        "split_membership_violations": len(violations),
        "violation_examples": violations[:10],
        "passed": len(rows) == expected_rows and complete_subjects == len(labels) and duplicates == 0 and not violations,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cfg = Config(root=root, folds=args.folds, repeats=args.repeats)
    labels = load_labels(root / cfg.label_file)
    results = [
        audit_file(resolve(root, path), labels, cfg, args.folds, args.repeats, args.repeat_offset)
        for path in args.predictions
    ]
    passed = all(bool(result["passed"]) for result in results)
    lines = [
        "# Crossfit 无直接外层泄漏审计",
        "",
        f"- 评估协议：subject-level {args.folds}-fold × {args.repeats} repeats，repeat offset={args.repeat_offset}。",
        f"- 总体结论：{'PASS' if passed else 'FAIL'}。",
        "- 审计范围：逐条预测验证被试属于对应外层 test，且不属于同一 split 的 train/val。",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## `{Path(str(result['path'])).name}`",
                f"- 行数：{result['rows']} / 期望 {result['expected_rows']}。",
                f"- 唯一被试：{result['unique_subjects']}；每人恰好 {args.repeats} 条预测的被试数：{result['subjects_with_expected_predictions']}。",
                f"- subject-repeat 重复行：{result['duplicate_subject_repeat_rows']}。",
                f"- split 成员关系违规：{result['split_membership_violations']}。",
                f"- 结论：{'PASS' if result['passed'] else 'FAIL'}。",
                "",
            ]
        )
        for example in result["violation_examples"]:
            lines.append(f"- 违规示例：{example}")
    lines.extend(
        [
            "## 边界",
            "- 该审计证明预测生成阶段没有把对应外层测试被试放入训练或验证集合。",
            "- PCA、覆盖率过滤、稳定选择、分数方向和神经网络拟合均由 crossfit 脚本在外层训练数据上重新执行。",
            "- 该审计不能消除在同一 40 人上反复比较不同算法造成的实验级 meta-selection bias；最终泛化仍需锁定方案后使用新增被试验证。",
        ]
    )
    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
