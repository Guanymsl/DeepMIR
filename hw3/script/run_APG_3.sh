#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for momentum in -0.9 -0.7 -0.5 -0.3 0 0.3 0.5 0.7 0.9; do
    echo "========================================"
    echo "Running APG scale=9 momentum=${momentum}"
    echo "========================================"

    python3 apg.py \
        --scale 9 \
        --eta 0 \
        --momentum "$momentum" \
        --norm-threshold 0

    python3 evaluate.py --pred "apg_w9_eta0_m${momentum}_n0"

    echo "Finished APG scale=9 momentum=${momentum}"
done

echo "Experiment 3 completed."
