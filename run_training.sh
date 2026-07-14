#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="results/retrained"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" scripts/run_signal_pca_nn_crossfit.py \
  --branch ranking --repeats 10 --repeat-offset 0 --device "${DEVICE}" \
  --out "${OUT_DIR}/auto_signal_pca_crossfit10_predictions.csv" --overwrite

"${PYTHON_BIN}" scripts/run_signal_pca_nn_crossfit.py \
  --branch classification --repeats 10 --repeat-offset 0 --device "${DEVICE}" \
  --out "${OUT_DIR}/auto_signal_pca_subspace15_crossfit10_predictions.csv" --overwrite

"${PYTHON_BIN}" scripts/run_signal_pca_nn_crossfit.py \
  --branch ranking --repeats 10 --repeat-offset 10 --device "${DEVICE}" \
  --out "${OUT_DIR}/auto_signal_pca_crossfit10_offset10_predictions.csv" --overwrite

"${PYTHON_BIN}" scripts/run_signal_pca_nn_crossfit.py \
  --branch classification --repeats 10 --repeat-offset 10 --device "${DEVICE}" \
  --out "${OUT_DIR}/auto_signal_pca_subspace15_crossfit10_offset10_predictions.csv" --overwrite

echo "Training completed. Run: bash run_reproduce_results.sh ${OUT_DIR}"
