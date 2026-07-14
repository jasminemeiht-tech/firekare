# Repeat Seed Holdout 复核

## 固定规则

- 排序：每名被试的 raw score 使用 Tukey biweight，标准常数 `c=4.685`。
- 分类：`min(mean(effect_score), mean(ecdf_evidence_score)) >= 0.5`。
- 规则由 repeats 0-9 的开发分析锁定；repeats 10-19 仅用于种子留出复核，没有再修改公式或阈值。
- 这仍然是同一 40 名被试上的算法稳定性检查，不是独立被试外部验证。

## 结果

| 划分 | 排序 AUROC | 排序 AUPRC | Accuracy | Balanced accuracy | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|
| development_repeats_0_9 | 0.9248 | 0.8729 | 0.8750 | 0.8574 | 26/3/2/9 |
| seed_holdout_repeats_10_19 | 0.8715 | 0.7840 | 0.8000 | 0.7492 | 25/4/4/7 |
| pooled_repeats_0_19 | 0.9028 | 0.8458 | 0.8000 | 0.7492 | 25/4/4/7 |

## 结论

- 开发 seeds 的 `0.9248/0.8750` 在新 seeds 上降为 `0.8715/0.8000`，没有复现。
- 合并 20 repeats 后为 `0.9028/0.8000`，说明前 10 repeats 的峰值对随机划分敏感。
- repeats 0-9 的全部 5-8 repeat 子集中，accuracy 从未达到 `0.875`；该峰值只在恰好使用全部 10 次时出现。
- 因此 `0.9248/0.8750` 应降级为未复现的开发性峰值，不能作为论文确认性主结果。

## 审计

- 原 repeats 0-9 两个源文件的成员审计：`reports/auto_signal_pca_robust_consensus_leakage_audit.md`，PASS。
- repeats 10-19 排序审计：`reports/auto_signal_pca_crossfit10_offset10_leakage_audit.md`，PASS。
- repeats 10-19 分类审计：`reports/auto_signal_pca_subspace15_crossfit10_offset10_leakage_audit.md`，PASS。
- PASS 只表示对应外层测试被试没有进入同一模型的训练/验证；不能消除同一 40 人上的开发偏差。
