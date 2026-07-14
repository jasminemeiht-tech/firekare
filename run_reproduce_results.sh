#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_DIR="${1:-results/predictions}"

"${PYTHON_BIN}" scripts/analyze_repeat_seed_holdout.py \
  --development-ranking "${SOURCE_DIR}/auto_signal_pca_crossfit10_predictions.csv" \
  --development-classification "${SOURCE_DIR}/auto_signal_pca_subspace15_crossfit10_predictions.csv" \
  --holdout-ranking "${SOURCE_DIR}/auto_signal_pca_crossfit10_offset10_predictions.csv" \
  --holdout-classification "${SOURCE_DIR}/auto_signal_pca_subspace15_crossfit10_offset10_predictions.csv" \
  --out results/summaries/reproduced_repeat_seed_holdout.csv \
  --subset-out results/summaries/reproduced_repeat_subset_sensitivity.csv \
  --report-out docs/REPRODUCED_RESULT.md \
  --overwrite

"${PYTHON_BIN}" scripts/audit_crossfit_no_leakage.py \
  --predictions "${SOURCE_DIR}/auto_signal_pca_crossfit10_predictions.csv" \
  --predictions "${SOURCE_DIR}/auto_signal_pca_subspace15_crossfit10_predictions.csv" \
  --folds 5 --repeats 10 --repeat-offset 0 \
  --out results/audits/reproduced_development_leakage_audit.md

"${PYTHON_BIN}" scripts/audit_crossfit_no_leakage.py \
  --predictions "${SOURCE_DIR}/auto_signal_pca_crossfit10_offset10_predictions.csv" \
  --predictions "${SOURCE_DIR}/auto_signal_pca_subspace15_crossfit10_offset10_predictions.csv" \
  --folds 5 --repeats 10 --repeat-offset 10 \
  --out results/audits/reproduced_holdout_leakage_audit.md

echo "Reproduced summaries and audits successfully."

