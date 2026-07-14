# 文件索引

## 根目录

- `README.md`：快速入口、最终结论和复现边界。
- `requirements.txt`：Python 依赖。
- `run_reproduce_results.sh`：从已保存预测重新聚合并审计。
- `run_training.sh`：从 658 维特征重新训练四个分支。
- `标签.xlsx`：40 名被试的标签和动作时间窗。

## `scripts/`

- `build_domain_landing_features.py`：原始数据 -> 658 维领域特征。
- `run_signal_pca_nn_crossfit.py`：精简的严格 signal-PCA + spline NN 训练入口。
- `analyze_repeat_seed_holdout.py`：开发/种子复核/pooled 固定聚合。
- `build_robust_consensus_dual_output.py`：Tukey biweight 和双输出帮助函数。
- `audit_crossfit_no_leakage.py`：逐条预测的外层成员审计。

## `features/`

- `domain_landing_features.csv`：40 行被试、2 列索引/标签和 658 个候选特征。

## `results/predictions/`

- `auto_signal_pca_crossfit10_predictions.csv`：开发 repeats 0-9 排序分支。
- `auto_signal_pca_subspace15_crossfit10_predictions.csv`：开发 repeats 0-9 分类分支。
- `auto_signal_pca_crossfit10_offset10_predictions.csv`：repeats 10-19 排序分支。
- `auto_signal_pca_subspace15_crossfit10_offset10_predictions.csv`：repeats 10-19 分类分支。

每个文件均为 400 行，40 名被试每人 10 条外层未见预测。

## `results/summaries/` 和 `results/audits/`

- 保存各分支汇总、repeat seed holdout 结果、repeat 子集敏感性和 PASS 审计文档。

## `docs/`

- `METHOD_AND_CONCLUSION.md`：建议优先阅读的方法和结论。
- `DATA_PLACEMENT.md`：原始数据放置方式。
- `auto_signal_pca_repeat_seed_holdout.md`：固定规则的 seed 复核报告。
- `leakage_safe_crossfit_signal_pca_nn_report.md`：原严格交叉拟合报告。
- `paper_ready_final_scheme.md`：论文口径的方案说明。

