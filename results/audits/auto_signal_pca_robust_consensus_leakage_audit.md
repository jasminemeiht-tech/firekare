# Crossfit 无直接外层泄漏审计

- 评估协议：subject-level 5-fold × 10 repeats。
- 总体结论：PASS。
- 审计范围：逐条预测验证被试属于对应外层 test，且不属于同一 split 的 train/val。

## `auto_signal_pca_crossfit10_predictions.csv`
- 行数：400 / 期望 400。
- 唯一被试：40；每人恰好 10 条预测的被试数：40。
- subject-repeat 重复行：0。
- split 成员关系违规：0。
- 结论：PASS。

## `auto_signal_pca_subspace15_crossfit10_predictions.csv`
- 行数：400 / 期望 400。
- 唯一被试：40；每人恰好 10 条预测的被试数：40。
- subject-repeat 重复行：0。
- split 成员关系违规：0。
- 结论：PASS。

## 边界
- 该审计证明预测生成阶段没有把对应外层测试被试放入训练或验证集合。
- PCA、覆盖率过滤、稳定选择、分数方向和神经网络拟合均由 crossfit 脚本在外层训练数据上重新执行。
- 该审计不能消除在同一 40 人上反复比较不同算法造成的实验级 meta-selection bias；最终泛化仍需锁定方案后使用新增被试验证。
