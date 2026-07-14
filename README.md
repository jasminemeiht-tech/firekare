# Leakage-Safe Signal-PCA Dual-Output Package

本包是 40 名被试多模态生物力学二分类的可转交版本，只保留当前严格主线：

`658 维领域特征 -> 训练折内 signal-PCA -> 稳定特征选择 -> spline_logit 神经网络 -> subject-level repeated crossfit -> 固定双输出聚合`

## 最终结论

- 更保守的 pooled repeats 0-19 结果：
  - 排序输出：AUROC `0.9028`，AUPRC `0.8458`。
  - 分类输出：Accuracy `0.8000`，Balanced Accuracy `0.7492`。
- AUROC 和 Accuracy 来自两个不同输出，不能写成同一概率分数同时达到 `0.903/0.800`。
- 开发 repeats 0-9 的 `0.9248/0.8750` 在 repeats 10-19 上没有复现，不作为最终泛化结论。
- 每条外层预测均由未见过该被试的模型生成；成员审计为 PASS。
- 该队列已被反复用于方法开发，仍存在 meta-selection bias，最终确证需要新被试或外部队列。

完整方法与结论见 [`docs/METHOD_AND_CONCLUSION.md`](docs/METHOD_AND_CONCLUSION.md)。

## 快速复算已保存结果

```bash
python -m pip install -r requirements.txt
bash run_reproduce_results.sh
```

该命令使用包内已保存的 4 个外层预测文件，重新执行固定聚合和成员审计。

## 从原始数据重新训练

1. 将原始目录放到包根目录：

```text
泄漏安全包/
├── 标签.xlsx
├── 肌电/
└── 运动力/
```

2. 生成 658 维特征：

```bash
python scripts/build_domain_landing_features.py --root . \
  --out features/domain_landing_features.csv --overwrite
```

3. 训练四个分支快照（ranking/classification x repeats 0-9/10-19）：

```bash
bash run_training.sh
```

默认使用 CPU。如果已安装可用的 CUDA 版 PyTorch，可使用：

```bash
DEVICE=cuda bash run_training.sh
```

4. 聚合和审计：

```bash
bash run_reproduce_results.sh results/retrained
```

## 重要复现边界

- `results/predictions/` 中的文件是当前数值结论的权威源预测，可精确复算聚合指标。
- 包内精简训练脚本实现了锁定算法，但历史开发期间原训练代码有过演进，加上 CUDA/BLAS 数值差异，从头训练不承诺逐位匹配源预测。
- 不要根据接收方重跑后的这 40 人结果继续修改特征、阈值或聚合规则，否则会继续增加选择偏差。
