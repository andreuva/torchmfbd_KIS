#!/usr/bin/env bash
# Reconstruct both CRISP scans in parallel, one scan per GPU, detached from the terminal.
#
#   ./run_gris.sh                 science1 on GPU 0 and science2 on GPU 2 (edit the pairs below)
#   ./run_gris.sh --crop_size 1408   extra arguments are passed on to gris_momfbd.py
#   TAG=v2 ./run_gris.sh --config gris_momfbd_v2.yaml --crop_size 1408 --simultaneous_seq 441 \
#       --output_dir results_momfbd_v2
#                                 TAG names the logs logs/<scan>_gpu<N>_<TAG>.log; the script
#                                 refuses to start if a log of that name exists
#
# Re-running with a new TAG resumes: steps whose output file exists are skipped. Run one step by hand first
# (python gris_momfbd.py --scan science1 --gpu 0 --steps 1 --output_dir results_test) so that
# the KL basis in basis/ exists before two processes could try to write it at the same time.
set -uo pipefail
cd "$(dirname "$0")"              # basis/ cache and logs/ live next to the scripts
PY=/dat/andreuva/gpu/miniconda/envs/torchmfbd/bin/python
mkdir -p logs

PAIRS="science1:0 science2:2"     # scan:gpu
for pair in $PAIRS; do            # never overwrite the logs of an earlier run
    log="logs/${pair%%:*}_gpu${pair##*:}${TAG:+_$TAG}.log"
    if [ -e "$log" ]; then echo "$log exists: set TAG=<name> to keep the logs of the earlier run"; exit 1; fi
done

for pair in $PAIRS; do
    scan=${pair%%:*}
    gpu=${pair##*:}
    log="logs/${scan}_gpu${gpu}${TAG:+_$TAG}.log"
    echo "=== $scan on GPU $gpu, log: $log ($(date)) ==="
    nohup "$PY" gris_momfbd.py --scan "$scan" --gpu "$gpu" "$@" > "$log" 2>&1 &
done
wait
echo "=== both scans finished ($(date)) ==="
