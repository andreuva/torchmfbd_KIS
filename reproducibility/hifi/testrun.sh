#!/usr/bin/env bash
# Run a quick HiFI+ test on the first 20260828 observation file.
# Usage: ./run_all_hifi_gpu.sh [gpu_id]
set -euo pipefail

GPU_ID="$1"
NUM_SHARDS="$2"
PROJECT_DIR="/dat/xenosi/torchmfbd"
DATA_DIR="/dat/schliche/gregor/hifiplus/hifiplus2/20260828"
CONFIG="$PROJECT_DIR/reproducibility/hifi/hifi_momfbd_gpu_improved.yaml"
OUTPUT_DIR="$PROJECT_DIR/reproducibility/hifi/results_momfbd/20260828"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== HiFI+ test on 20260828 (requested GPU $GPU_ID) ==="

/dat/xenosi/miniconda/envs/torchmfbd_fixed/bin/python \
    reproducibility/hifi/hifi_momfbd_batch_gpu.py \
    --input_dir "$DATA_DIR" \
    --pattern "hifiplus2_20260828_081713_sd.fts" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --gpu "$GPU_ID" \
    --num_shards "$NUM_SHARDS" \
    --simultaneous_seq 200\
    --no_resume \
    --limit 1 \
    
echo "=== Finished; output directory: $OUTPUT_DIR ==="

#higher stride means less overlap, which means less memory usage, but also more artifacts.