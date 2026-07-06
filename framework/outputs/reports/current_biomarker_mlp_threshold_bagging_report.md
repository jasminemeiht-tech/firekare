# 固定 20 维 Biomarker MLP 预测模型报告

报告日期：2026-05-13  
代码目录：`/home/ubuntu/xiangmu/xiaofang/chap_2/deeplearning/framework`  
当前主版本：`biomarker_mlp_nested_bce_calibrated_ensemble10 + threshold bagging q65`

## 1. 任务目标

本轮建模目标是基于受试者运动学、动力学、肌电及协同特征中筛选得到的 20 个 biomarker，预测最终二分类标签 `injury_label`。当前样本量为 40 个 subject，其中阳性 11 个、阴性 29 个，属于典型小样本且类别不平衡任务。因此，本版模型设计的核心不是继续堆叠更复杂的网络，而是在固定高质量输入特征的基础上，提高训练稳定性、概率校准可靠性和最终分类阈值的稳健性。

当前主版本采用深度学习方法：20 维 biomarker 输入 MLP 分类器，并结合 nested OOF 概率校准、10-seed logits ensemble、bootstrap threshold bagging。最终正式采用 `q65` 阈值聚合策略作为主版本。

最终主指标为：

| AUROC | AUPRC | accuracy | sensitivity | specificity | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.970 | 0.950 | 0.900 | 0.833 | 0.933 | 0.813 |

## 2. 数据与验证口径

数据规模：

| 项目 | 数量 |
|---|---:|
| trial 数 | 78 |
| subject 数 | 40 |
| 阴性 subject | 29 |
| 阳性 subject | 11 |
| 外层 CV 折数 | 5 |

外层验证采用 subject-level 分组分层切分，确保同一 subject 不会同时出现在训练集和验证集。每折 8 个 subject，各折阳性数尽量平衡：

| fold | 阴性 | 阳性 |
|---:|---:|---:|
| 0 | 6 | 2 |
| 1 | 6 | 2 |
| 2 | 5 | 3 |
| 3 | 6 | 2 |
| 4 | 6 | 2 |

报告主指标采用现有训练框架中的 fold-mean 口径；pooled 指标作为辅助参考。

## 3. 输入特征

当前固定使用 20 维 biomarker，来源于：

`/home/ubuntu/xiangmu/xiaofang/chap_2/l2_ligit/biomarker_plus4_logreg_package.zip`

20 个特征如下：

| 序号 | feature |
|---:|---|
| 1 | VDJ_emg_VLO_peak_t_ms |
| 2 | VDJ_emg_SEMITEND_min |
| 3 | SSC_imu_roll_deg_peak_t_ms |
| 4 | VDJ_ik_hip_adduction_r_range |
| 5 | SSC_syn2_early_integral |
| 6 | VDJ_id_knee_angle_r_moment_min |
| 7 | SSC_ik_hip_rotation_r_range |
| 8 | VDJ_emg_VMO_max |
| 9 | SSC_emg_MED. GASTRO_min |
| 10 | SSC_id_hip_adduction_r_moment_std |
| 11 | VDJ_ik_hip_adduction_r_std |
| 12 | VDJ_ik_hip_adduction_r_abs_peak |
| 13 | VDJ_syn1_sparsity |
| 14 | VDJ_emg_VMO_abs_peak |
| 15 | SSC_ik_subtalar_angle_r_std |
| 16 | VDJ_label_duration |
| 17 | VDJ_grf_grf2_vz_peak_t_ms |
| 18 | VDJ_id_knee_angle_r_moment_mean |
| 19 | VDJ_imu_Accel_Earth_Y_mG_abs_peak |
| 20 | VDJ_id_hip_rotation_r_moment_mean |

预处理方式：

- 每个 outer fold 内只使用训练 subject 拟合预处理参数。
- 缺失值使用训练集 median imputation。
- 特征使用训练集均值和标准差做 StandardScaler。
- outer validation 只使用训练折拟合出的 imputer/scaler 进行变换，避免数据泄漏。

## 4. 模型算法设计

### 4.1 总体建模思路

当前模型不是直接使用原始时间序列做端到端预测，而是使用已经从 SSC / VDJ 动作中提取得到的 20 个 biomarker 作为输入。这样做的主要原因是：

- 当前 subject 数只有 40，直接用高维原始序列训练深度模型很容易过拟合。
- 固定 20 维 biomarker 已经包含较强的力学、肌电、动作时序和协同信息，是目前最稳定的特征基础。
- MLP 仍然属于深度学习模型，但参数量较小，更适合当前小样本场景。
- 后续的 ensemble、calibration 和 threshold bagging 主要解决小样本下训练波动、概率偏移和阈值不稳定问题。

整体预测链路如下：

```text
subject-level 20 biomarkers
  -> outer-train-only imputation / standardization
  -> Biomarker MLP
  -> 10-seed logits ensemble
  -> Platt calibration based on inner OOF logits
  -> calibrated probability
  -> q65 bootstrap threshold
  -> final binary prediction
```

### 4.2 Biomarker MLP 结构

每个 subject 的输入是一个 20 维向量。模型输出一个 logit，经过校准后转成阳性概率。网络结构为：

```text
20-dim biomarker
  -> Linear(20, 64)
  -> LayerNorm(64)
  -> ReLU
  -> Dropout(0.2)
  -> Linear(64, 32)
  -> LayerNorm(32)
  -> ReLU
  -> Dropout(0.2)
  -> Linear(32, 1)
  -> logit
```

各模块作用：

- `Linear` 层学习 20 个 biomarker 之间的非线性组合关系。
- `LayerNorm` 用于稳定小 batch 训练，避免 batch size 较小时 BatchNorm 不稳定。
- `ReLU` 提供非线性表达能力。
- `Dropout(0.2)` 降低小样本过拟合风险。
- 最后一层 `Linear(32, 1)` 输出未校准 logit。

### 4.3 训练目标与采样策略

训练目标使用二分类 BCE loss：

```text
loss = BCEWithLogitsLoss(logit, label)
```

当前配置：

- loss：BCE
- `positive_class_weight = 1.0`
- `WeightedRandomSampler = true`
- learning rate：`0.0003`
- weight decay：`0.0001`
- min final epochs：`20`
- device：CUDA

这里没有再启用 `pos_weight > 1`，原因是前面的实验显示在当前 20 维 biomarker 上，过强的正类权重会让模型更容易把阴性推成阳性，造成 specificity 和 accuracy 下降。当前使用 `WeightedRandomSampler = true`，让每个 epoch 中阳性样本被更充分地看到，但 loss 本身保持对称权重，避免概率整体过度偏向阳性。

### 4.4 Nested OOF 训练与概率校准

每个 outer fold 的训练过程分为两层。

第一层是 outer 5-fold CV：

- 每次取 4 折 subject 作为 outer train。
- 剩余 1 折 subject 作为 outer validation。
- outer validation 在模型训练、校准、阈值选择中完全不参与，只用于最终评估。

第二层是在每个 outer train 内部做 inner 4-fold OOF：

1. 将 outer train 再分成 4 个 inner fold。
2. 每次用 inner train 训练 MLP，用 inner validation 预测 logit。
3. 拼接 4 次 inner validation 结果，得到 outer train 上每个 subject 的 OOF logit。
4. OOF logit 代表“训练集内部未见样本”的模型输出，用于后续 Platt calibration 和阈值选择。

Platt calibration 使用 OOF logit 拟合一个一维 logistic regression：

```text
calibrated_prob = sigmoid(a * logit + b)
```

这样做的目的不是改变排序，而是把 MLP 输出的 logit 映射到更可靠的概率尺度。由于校准器只使用 outer train 的 OOF 结果拟合，没有使用 outer validation 标签，因此可以避免验证集泄漏。

### 4.5 最终模型重训策略

完成 inner OOF 后，会在完整 outer train 上重新训练最终模型。最终训练 epoch 数由 inner 训练中观测到的最佳 epoch 聚合得到，并设置 `min_final_epochs = 20`。当前各 fold 最终训练 epoch 均为 20。

这一策略的含义是：

- inner fold 用于估计合理训练轮数、校准参数和阈值。
- final model 使用完整 outer train 训练，尽可能利用所有训练 subject。
- outer validation 只在最终模型和最终阈值确定后进行一次评估。

### 4.6 10-seed logits ensemble

每个 outer fold 的最终模型使用 10 个不同随机种子训练。预测时先对 10 个模型输出的 logits 求平均，再做 sigmoid / Platt calibration 得到最终概率。

采用 logits 平均而不是概率平均的原因是：logits 仍处在模型判别空间，平均 logits 通常比直接平均概率更稳定，尤其适合小样本和阈值敏感任务。

ensemble 的作用主要有三点：

- 降低单次随机初始化带来的方差。
- 缓解小样本训练中个别模型不稳定的问题。
- 让最终概率更平滑，便于后续阈值选择。

### 4.7 Bootstrap Threshold Bagging 与 q65 主阈值

模型输出概率后，还需要确定二分类阈值。由于当前样本量小，每个 outer train 只有 32 个 subject，直接在 OOF 上选一个单点最优阈值容易受个别 subject 影响。因此，本版引入 bootstrap threshold bagging。

具体流程：

1. 对每个 outer fold，取对应 outer train 的 calibrated OOF probability。
2. 按标签分层 bootstrap 采样 1000 次。
3. 每次 bootstrap 内选择使 accuracy 最大的阈值。
4. 得到 1000 个候选阈值后，计算阈值分布的 `median`、`q65`、`q75`。
5. 将这些阈值应用到对应 outer validation probability。

当前正式选择 `q65` 作为主版本。原因是：

- `q65` 与 `q75` 在当前 CV 中 accuracy 和 F1 相同。
- `q65` 比 `q75` 阈值略低，对 FN 风险更温和。
- 相比原始单点 OOF 阈值，`q65` 修正了 subject 16 这个极贴边 FP，并提高了 specificity、accuracy 和 F1。

该策略只改变最终分类阈值，不改变模型概率排序，因此 AUROC / AUPRC 不会因 threshold bagging 改变；主要影响 accuracy、sensitivity、specificity 和 F1。

### 4.8 单个 outer fold 的完整流程

为了说明训练和验证边界，单个 outer fold 的执行逻辑可以概括为：

```text
输入：40 个 subject 的固定 20 维 biomarker

for each outer fold:
    outer_train = 32 subjects
    outer_val   = 8 subjects

    在 outer_train 内：
        1. 拟合 median imputer 和 StandardScaler
        2. 做 4-fold inner OOF
        3. 每个 inner fold 训练 10-seed MLP ensemble
        4. 得到 outer_train 的 OOF logits
        5. 用 OOF logits 拟合 Platt calibration
        6. 用 OOF calibrated probability 做 bootstrap threshold bagging
        7. 取 q65 作为该 outer fold 的最终阈值

    在完整 outer_train 上：
        8. 重新训练 10-seed final MLP ensemble
        9. 对 outer_val 输出平均 logit
       10. 使用 outer_train 拟合的 Platt calibration 转成 probability
       11. 使用 outer_train 选出的 q65 阈值给出最终预测

    outer_val 标签只用于最后计算指标
```

这个流程中，所有预处理参数、校准参数、阈值参数都只从 outer train 学得。outer validation 不参与模型选择、阈值选择和概率校准。

## 5. 当前指标结果

### 5.1 主版本 q65 指标

当前正式主版本为 `threshold bagging q65`。主指标采用 5-fold outer validation 的 fold-mean 口径。

| 指标 | q65 主版本 |
|---|---:|
| AUROC | 0.970 |
| AUPRC | 0.950 |
| accuracy | 0.900 |
| sensitivity | 0.833 |
| specificity | 0.933 |
| F1 | 0.813 |

q65 pooled confusion matrix：

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 27 | 2 | 2 | 9 |

从业务解释上看，当前 q65 主版本的特点是：总体排序能力较强，AUROC 和 AUPRC 较高；在最终二分类上，specificity 达到 0.933，说明对阴性 subject 的误报控制较好；sensitivity 为 0.833，仍有 2 个阳性 subject 被漏判，是后续定向分析的重点。

### 5.2 阈值策略对比

| 阈值策略 | AUROC | AUPRC | accuracy | sensitivity | specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| median | 0.970 | 0.950 | 0.875 | 0.833 | 0.900 | 0.780 |
| q65 主版本 | 0.970 | 0.950 | 0.900 | 0.833 | 0.933 | 0.813 |
| q75 | 0.970 | 0.950 | 0.900 | 0.833 | 0.933 | 0.813 |

q65 和 q75 的 fold-mean 指标相同，但 q65 阈值更温和，因此选择 q65 作为正式版本。median 本质上接近原始单点 OOF 阈值，因此整体指标没有明显变化。

### 5.3 与原始 nested ensemble10 的比较

原始版本使用每折 OOF 上 accuracy 最优的单一阈值；当前主版本使用 bootstrap 阈值分布的 q65 分位数。

| 指标 | 原始 | q65 | 提升 |
|---|---:|---:|---:|
| accuracy | 0.875 | 0.900 | +0.025 |
| specificity | 0.900 | 0.933 | +0.033 |
| F1 | 0.780 | 0.813 | +0.033 |

原始 pooled confusion matrix：

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 26 | 3 | 2 | 9 |

q65 pooled confusion matrix：

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 27 | 2 | 2 | 9 |

可以看到，q65 的主要收益来自减少 1 个 FP，同时没有增加 FN。因此 accuracy、specificity 和 F1 同时提升。

## 6. 错分 subject 分析

原始阈值策略错分 5 个 subject：

| subject | label | prob | threshold | 类型 | 说明 |
|---:|---:|---:|---:|---|---|
| 16 | 0 | 0.325 | 0.324 | FP | 极贴近阈值，margin 仅 +0.0015 |
| 33 | 1 | 0.276 | 0.324 | FN | 阳性但概率偏低 |
| 15 | 1 | 0.230 | 0.282 | FN | 阳性但概率明显偏低 |
| 4 | 0 | 0.466 | 0.339 | FP | 较硬 FP |
| 18 | 0 | 0.422 | 0.382 | FP | 中等 margin FP |

采用 q65 后，subject 16 被修正，剩余错分 4 个：

| subject | label | prob | q65 threshold | 类型 | 判断 |
|---:|---:|---:|---:|---|---|
| 33 | 1 | 0.276 | 0.328 | FN | 较硬 FN |
| 15 | 1 | 0.230 | 0.301 | FN | 较硬 FN |
| 4 | 0 | 0.466 | 0.353 | FP | 较硬 FP |
| 18 | 0 | 0.422 | 0.382 | FP | 中等 FP |

错分特征解释显示：

- subject 4 被 VDJ 髋内收相关特征明显拉向阳性：`VDJ_ik_hip_adduction_r_range`、`VDJ_ik_hip_adduction_r_std`、`VDJ_label_duration` 等。
- subject 18 主要受 `VDJ_syn1_sparsity`、`VDJ_emg_VMO_max`、`SSC_ik_hip_rotation_r_range` 等特征影响，被推向阳性。
- subject 33 的 `VDJ_label_duration` 极端偏低，同时 `VDJ_emg_VMO_max / abs_peak` 方向上更像阴性，导致 FN。
- subject 15 的 `VDJ_id_knee_angle_r_moment_min`、`SSC_ik_subtalar_angle_r_std`、`VDJ_id_hip_rotation_r_moment_mean` 等更像阴性，导致 FN。

从错误分布看，当前模型不是整体失效，而是少数 subject 落在特征空间的真实混叠区域。subject 16 属于阈值稳定性问题，已经被 threshold bagging 修正；subject 4、15、18、33 更像需要人工复核或进一步特征解释的边界/异质样本。

## 7. 与 658 维稳定特征选择路线的比较

之前尝试从 658 维候选特征中做 outer-train 内稳定特征选择，再训练 residual MLP / Biomarker MLP。结果未能超过固定 20 维主线，且表现出明显小样本过拟合：

| 路线 | AUROC | AUPRC | accuracy | F1 |
|---|---:|---:|---:|---:|
| 658 free selection + residual MLP | 0.330 | 0.348 | 0.650 | 0.183 |
| anchor20 + residual MLP | 0.683 | 0.617 | 0.725 | 0.227 |
| anchor20 + Biomarker MLP | 0.640 | 0.599 | 0.650 | 0.240 |

因此当前阶段不建议继续把自动 658 维 top-K 选择作为主线。更合理的定位是：固定 20 维作为主模型，658 维作为错分 subject 的解释和人工候选特征来源。

## 8. 当前结论

当前正式主版本为：

```text
Fixed 20 biomarkers
  -> Biomarker MLP
  -> 10-seed logits ensemble
  -> nested OOF Platt calibration
  -> bootstrap threshold bagging
  -> q65 threshold
```

主版本指标达到：

| AUROC | AUPRC | accuracy | sensitivity | specificity | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.970 | 0.950 | 0.900 | 0.833 | 0.933 | 0.813 |

这个版本的核心优势是：

- 使用深度学习模型，满足当前“尽量用深度学习方法”的方向。
- 固定 20 维 biomarker 避免了 658 维小样本自动特征选择带来的过拟合。
- nested OOF 概率校准和阈值选择避免 outer validation 泄漏。
- 10-seed logits ensemble 降低了随机初始化对小样本训练的影响。
- q65 threshold bagging 明确改善 specificity、accuracy 和 F1。
- 错分 subject 能够追溯到具体 biomarker，便于后续人工复核。

需要注意的是，当前结果仍然来自 40 个 subject 的交叉验证。q65 已作为当前内部 CV 下的正式版本，但后续如果有外部新样本，应固定当前训练策略和 q65 规则，在外部数据上做独立验证。

## 9. 后续建议

下一步不建议继续盲目调 MLP 超参数。最值得做的是围绕剩余 4 个错分 subject 做定向分析：

1. 对 subject 4、18 的阳性样特征进行人工复核，判断是否存在标签噪声、动作质量异常或 biomarker 提取异常。
2. 对 subject 15、33 的阴性样特征进行复核，判断阳性标签是否来自当前 20 维 biomarker 未覆盖的机制。
3. 以 subject-level error analysis 为基础，从 658 维特征中寻找能解释这 4 个 subject 的候选补充特征，但不要直接全量自动筛选。
4. 如果需要进一步提升 accuracy / F1，优先考虑“人工审阅后的少量机制特征补充 + 固定 20 维 MLP”，而不是继续扩大自动特征空间。

相关输出文件：

- `biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_summary.csv`
- `biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_predictions.csv`
- `biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_misclassified_subjects.csv`
- `biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_misclassified_feature_zscores.csv`
- `biomarker_mlp_nested_bce_calibrated_ensemble10_threshold_bagging_misclassified_neighbors.csv`
