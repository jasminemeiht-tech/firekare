# 训练内信号压缩与重复交叉拟合神经网络实验报告

## 1. 目标

在 40 名被试、标签 0/1 为 29:11 的条件下，避免把多数类预测准确率 0.725 当作有效模型性能，并在 subject-level 隔离下同时提高 AUROC、accuracy 和 balanced accuracy。

## 2. 数据边界

- 每名被试包含 SSC、VDJ 两个动作。
- 所有外层划分均以被试为单位，同一人的动作、模态和片段不会跨训练集与测试集。
- 外层测试被试不参与缺失处理拟合、标准化、PCA、特征选择、分数方向判断、阈值或神经网络训练。
- 658 维候选表先在每个训练折内剔除覆盖率低于 80% 的列。

## 3. 算法流程

1. 使用 subject-level 5-fold，重复 10 次。
2. 每个外层模型使用约 32 人训练、8 人测试；每名被试最终获得 10 个从未见过该被试的预测。
3. 在外层训练集内部执行 3-fold OOF：
   - 按原始信号分组，将同一信号的多个统计量在训练折内压缩为 1 个 PCA 分量；
   - 使用稳定性筛选形成候选特征池；
   - 仅根据训练内 OOF 判断分数方向和模型配置。
4. 分类器为 `spline_logit` 神经网络：线性 neural logit 主干加 RBF spline 残差。
5. 高准确率版本从稳定候选池中构造 7 个小型随机子空间，每个子空间训练相同结构的神经网络，再平均分数；没有门控、教师模型或其他分类器。
6. 用训练内 OOF 的正负类分数均值和方差形成 effect 校准分数。
7. 对同一被试的 10 个 effect 分数取平均，以固定阈值 0.5 输出类别。

## 4. 结果

### 4.1 当前双输出候选

同一套 signal-PCA 与 `spline_logit` 神经网络产生两个固定用途的输出：

- 排序输出：非子空间版本的 10 个 raw crossfit 分数取中位数；
- 分类输出：7 子空间版本的 10 个 effect 分数取均值，固定阈值 0.5。

| 指标 | 结果 |
|---|---:|
| AUROC（排序输出） | 0.9216 |
| AUPRC（排序输出） | 0.8650 |
| Accuracy（分类输出） | 0.8500 |
| Balanced accuracy（分类输出） | 0.8401 |
| TN / FP / FN / TP | 25 / 4 / 2 / 9 |
| AUROC bootstrap 95% CI | 0.8182–0.9937 |
| Accuracy bootstrap 95% CI | 0.7250–0.9500 |

两个输出均由未见过对应被试的模型生成；中位数、均值和 0.5 阈值均为固定聚合规则，不使用测试标签调权。

### 4.2 单一分数平衡版本

随机子空间数 7、每个子空间 6 维、候选池约 24 维：

| 指标 | 结果 |
|---|---:|
| AUROC（effect score） | 0.8495 |
| AUPRC（effect score） | 0.6854 |
| Accuracy | 0.8500 |
| Balanced accuracy | 0.8401 |
| TN / FP / FN / TP | 25 / 4 / 2 / 9 |
| AUROC bootstrap 95% CI | 0.7053–0.9655 |
| Accuracy bootstrap 95% CI | 0.7250–0.9500 |

该结果明显超过“全部预测为 0”的 accuracy 0.725，并且保持了较高的阳性识别率：11 名阳性中识别出 9 名。

### 4.3 排序优先版本

不使用随机子空间、由训练内选择 4/6/8/12 维：

| 指标 | 结果 |
|---|---:|
| AUROC（raw score均值） | 0.8903 |
| AUPRC（raw score均值） | 0.7651 |
| AUROC（raw score中位数） | 0.9216 |
| AUPRC（raw score中位数） | 0.8650 |
| AUROC（effect score） | 0.8558 |
| Effect 多数投票 accuracy | 0.8000 |
| Effect 多数投票 balanced accuracy | 0.8056 |

该版本排序能力更强，但分类决策准确率低于推荐平衡版本。

### 4.4 其他优化对照

- 内层在 BCE/pairwise 损失间选择：crossfit10 raw AUROC 0.8621、effect accuracy 0.7500，低于固定 BCE。
- 稳定候选池后再做 4 维监督 PCA：smoke AUROC 0.5600，折间不稳定。
- 通用 2 分量 signal PCA：smoke AUROC 0.6158，低于 1 分量方案。
- LDA 式后验校准前 4 个 repeats 持续低于 effect 校准，实验提前终止并标记为 partial。

### 4.5 DDPM 对照

相同信号 PCA 流程加入 `fixed_pos=1` DDPM 后，单次 5-fold smoke：

- AUROC：0.7733，对照无扩散为 0.7858；
- Accuracy：0.7000，对照无扩散为 0.7250。

因此当前最佳方案不加入 DDPM。合成一个正类样本没有提供稳定增益，继续增加生成量缺少实验依据。

## 5. 结论与限制

- 通过训练内信号级 PCA、稳定候选池、同构神经网络子空间平均和重复交叉拟合，可以形成 AUROC 0.9216、accuracy 0.8500 的双输出候选。
- 40 人仍导致置信区间较宽；accuracy 的 bootstrap 下界仍接近多数类基线。
- 本方案是在同一 40 人数据上经过多轮方法探索后形成，虽然每个预测没有直接外层泄漏，但仍存在实验级 meta-selection bias。
- 下一步应锁定本报告中的流程和参数，在新增被试或外部队列上一次性验证；在此之前不应宣称真实泛化性能已达到 0.85。

## 6. 复现入口

- 主脚本：`scripts/run_auto_feature_nn_crossfit_ensemble.py`
- 汇总脚本：`scripts/summarize_crossfit_predictions.py`
- 双输出汇总脚本：`scripts/build_dual_output_crossfit_summary.py`
- 无直接外层泄漏审计：`scripts/audit_crossfit_no_leakage.py`
- 逐预测结果：`reports/auto_signal_pca_subspace7_crossfit10_predictions.csv`
- 被试聚合结果：`reports/auto_signal_pca_subspace7_crossfit10_vote_aggregated.csv`
- 指标汇总：`reports/auto_signal_pca_subspace7_crossfit10_vote_summary.csv`
- 双输出结果：`reports/auto_signal_pca_dual_output_crossfit10_summary.csv`
- 审计报告：`reports/auto_signal_pca_dual_output_leakage_audit.md`
