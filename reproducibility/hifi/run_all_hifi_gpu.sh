#!/usr/bin/env bash
# Runs one GPU's shard of the full HiFI+ MOMFBD batch across all datasets.
# Usage: run_all_hifi_gpu.sh <gpu_id> <num_shards>
#
# The configuration is hifi_momfbd.yaml, the same one hifi_momfbd.py uses for a
# single burst. hifi_momfbd_gpu.yaml and hifi_momfbd_gpu_improved.yaml are the
# older settings and are left only for reference - running with them reproduces
# the noisy reconstructions.
#
# Budget about 5 minutes per burst on an H100 (~1650 patches at stride 32).
set -uo pipefail

GPU_ID="$1"
NUM_SHARDS="$2"
DATA_ROOT="/dat/schliche/gregor/hifiplus/hifiplus2"
HIFI_DIR="/dat/xenosi/torchmfbd/reproducibility/hifi"

source /dat/xenosi/miniconda/etc/profile.d/conda.sh
conda activate torchmfbd_fixed
cd "$HIFI_DIR"

for dataset in 20260722 ; do
    echo "=== [GPU $GPU_ID] Starting dataset $dataset ($(date)) ==="
    python hifi_momfbd_batch_gpu.py \
        --input_dir "$DATA_ROOT/$dataset/" \
        --output_dir "$HIFI_DIR/results_momfbd/${dataset}_1" \
        --config "$HIFI_DIR/hifi_momfbd_gpu_off_limb_test.yaml" \
        --gpu "$GPU_ID" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$GPU_ID" \
        --simultaneous_seq 150 \
        --n_iterations 250 \
        --stride_size 42 \
        --start_time 07455 \
        --limb_mode off_limb \
        --no_resume \
        
    echo "=== [GPU $GPU_ID] Finished dataset $dataset ($(date)) ==="
done

echo "=== [GPU $GPU_ID] All datasets complete ==="
