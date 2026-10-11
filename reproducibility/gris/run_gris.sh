#!/usr/bin/env bash
# Reconstruct both CRISP scans in parallel, one scan per GPU, detached from the terminal.
#
#   ./run_gris.sh                 science1 on GPU 0 and science2 on GPU 2 (edit the pairs below)
#   ./run_gris.sh --crop_size 1408   extra arguments are passed on to gris_momfbd.py
#
# Re-running resumes: steps whose output file exists are skipped. Run one step by hand first
# (python gris_momfbd.py --scan science1 --gpu 0 --steps 1 --output_dir results_test) so that
# the KL basis in basis/ exists before two processes could try to write it at the same time.
set -uo pipefail
cd "$(dirname "$0")"              # basis/ cache and logs/ live next to the scripts
PY=/dat/andreuva/gpu/miniconda/envs/torchmfbd/bin/python
mkdir -p logs

for pair in science1:0 science2:2; do
    scan=${pair%%:*}
    gpu=${pair##*:}
    echo "=== $scan on GPU $gpu, log: logs/${scan}_gpu${gpu}.log ($(date)) ==="
    nohup "$PY" gris_momfbd.py --scan "$scan" --gpu "$gpu" "$@" > "logs/${scan}_gpu${gpu}.log" 2>&1 &
done
wait
echo "=== both scans finished ($(date)) ==="
