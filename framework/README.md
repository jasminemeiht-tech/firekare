# Deep Learning Framework

This directory contains a first-pass deep learning framework for injury-label
prediction on the `chap_2` dataset.

The framework is designed around the actual data constraints in this project:

- small cohort size (`subject` count is limited)
- repeated trials per subject
- multimodal time-series inputs (`EMG`, `GRF`, `IK`, `ID`)
- binary target (`injury_label`)
- optional diffusion-based augmentation, not diffusion-as-classifier

## Design Goals

1. Keep `subject_id` as the split unit.
2. Start from a strong baseline before using heavier models.
3. Support raw sequence modeling and handcrafted-feature baselines.
4. Put diffusion in an auxiliary role:
   - latent augmentation
   - denoising pretraining
   - representation learning

## Recommended Workflow

1. Build a normalized trial manifest:

```bash
python -m src.build_manifest --config config.yaml
```

2. Inspect the generated manifest:

- `outputs/manifests/trials_manifest.csv`
- `outputs/manifests/split_summary.json`

3. Run a baseline feature model:

```bash
python -m src.train --config config.yaml --stage baseline
```

4. Run the multimodal sequence model:

```bash
python -m src.train --config config.yaml --stage sequence
```

5. Enable diffusion augmentation after the non-diffusion pipeline is stable:

```bash
python -m src.train --config config.yaml --stage sequence --use-diffusion
```

## Directory Layout

```text
framework/
├── README.md
├── config.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── build_manifest.py
│   ├── constants.py
│   ├── data_io.py
│   ├── dataset.py
│   ├── diffusion.py
│   ├── manifest.py
│   ├── models.py
│   ├── splits.py
│   └── train.py
└── outputs/
    ├── manifests/
    ├── checkpoints/
    ├── logs/
    └── reports/
```

## Modeling Strategy

### Stage 1: Baseline

Use existing tabular features first:

- `new_analysis/outputs/features/core_features.csv`
- `xiazhijigu/outputs/features/joint_features.csv`
- `jirouxietong/outputs/features/synergy_features.csv`

Candidate baseline models:

- logistic regression
- random forest
- xgboost / lightgbm

### Stage 2: Sequence Classifier

Use aligned windows from raw signals:

- `EMG`: 8 channels
- `GRF`: summed / selected force channels
- `IK`: hip, knee, ankle angles
- `ID`: hip, knee, ankle moments

Recommended first model:

- two-branch 1D CNN encoder
- one branch for `EMG`
- one branch for `mechanics`
- `action` embedding as a condition input
- fusion head for binary classification

### Stage 3: Diffusion-Assisted Training

Use diffusion only after Stage 1 and 2 are stable.

Recommended roles:

- latent augmentation
- denoising pretraining on unlabeled trial windows
- class-conditional synthetic samples inside training folds only

Not recommended:

- direct diffusion classifier on raw high-dimensional inputs
- random trial-level splits
- training diffusion before a trustworthy baseline exists

## Current Assumptions

- label prediction target is `injury_label`
- the evaluation split unit is `subject_id`
- raw file paths in historical metadata may contain stale prefixes and must be
  normalized to the current workspace
- the default raw-data root on this machine is:

`/home/ubuntu/xiangmu/xiaofang/chap_2/data`

## Next Practical Step

The first useful artifact is the normalized manifest. Once that file is built,
we can connect preprocessing, dataloaders, and training without relying on the
older machine-specific paths.
