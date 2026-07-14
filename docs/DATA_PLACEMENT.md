# 数据放置说明

包内已包含：

- `标签.xlsx`：标签与 SSC/VDJ 时间窗。
- `features/domain_landing_features.csv`：已提取的 40 x 658 候选特征。
- `results/predictions/`：生成当前结论的源预测。

如果只需复算指标，不需要原始数据目录。

如果要从原始数据重新提取特征，请将以下两个目录放在包根目录：

```text
肌电/
运动力/
```

完整结构：

```text
leakage_safe_signal_pca_dual_output_package/
├── 标签.xlsx
├── 肌电/
├── 运动力/
├── features/
├── results/
├── scripts/
└── src/
```

特征提取命令：

```bash
python scripts/build_domain_landing_features.py --root . \
  --out features/domain_landing_features.csv --overwrite
```

原始文件命名和采样格式参见 `docs/原始数据说明.txt`。

