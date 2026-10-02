#!/usr/bin/env bash
# Plot the MOMFBD results for the configured dataset.
# Run with: ./run_all_hifi_plots.sh
set -euo pipefail

DATASET="20260820"
DATA_ROOT="/dat/schliche/gregor/hifiplus/hifiplus2"
HIFI_DIR="/dat/xenosi/torchmfbd/reproducibility/hifi"

source /dat/xenosi/miniconda/etc/profile.d/conda.sh
conda activate torchmfbd_fixed
cd "$HIFI_DIR"

python plot_momfbd_batch_result.py \
    --results_dir "$HIFI_DIR/results_momfbd/${DATASET}_test" \
    --raw_dir "$DATA_ROOT/$DATASET" \
    --output_dir "$HIFI_DIR/plots_momfbd/${DATASET}_test"\
    --overwrite \

