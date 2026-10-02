#!/usr/bin/env bash
# Run a quick HiFI+ test on the first 20260715 observation file.
# Usage: ./run_all_hifi_gpu.sh [gpu_id]
set -euo pipefail

GPU_ID="$1"
NUM_SHARDS="$2"
PROJECT_DIR="/dat/xenosi/torchmfbd"
DATA_DIR="/dat/xenosi/data/test_andreu_data/20260715"
CONFIG="$PROJECT_DIR/reproducibility/hifi/hifi_momfbd_gpu_off_limb_test.yaml"
OUTPUT_DIR="$PROJECT_DIR/reproducibility/hifi/results_momfbd/20260715_test"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== HiFI+ test on 20260715 (requested GPU $GPU_ID) ==="

/dat/xenosi/miniconda/envs/torchmfbd_fixed/bin/python \
    reproducibility/hifi/hifi_momfbd_batch_gpu.py \
    --input_dir "$DATA_DIR" \
    --pattern "hifiplus2_20260715_082902_sd.fts" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --gpu "$GPU_ID" \
    --num_shards "$NUM_SHARDS" \
    --simultaneous_seq 150 \
    --n_iterations 250 \
    --patch_size 96 \
    --stride_size 42 \
    --no_resume \
    --limb_mode off_limb \
    --limit 1 
    
echo "=== Finished; output directory: $OUTPUT_DIR ==="

#lower stride means higher overlap, which means higher memory usage, but also less artifacts.