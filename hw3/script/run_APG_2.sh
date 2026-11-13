#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for eta in 0.25 0.5 0.75 1; do
    echo "========================================"
    echo "Running APG scale=9 eta=${eta}"
    echo "========================================"

    python3 apg.py \
        --scale 9 \
        --eta "$eta" \
        --momentum -0.5 \
        --norm-threshold 0

    python3 evaluate.py --pred "apg_w9_eta${eta}_m-0.5_n0"

    echo "Finished APG scale=9 eta=${eta}"
done

echo "Experiment 2 completed."
