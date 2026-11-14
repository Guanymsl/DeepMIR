#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for mode in concise detailed planned; do
    echo "========================================"
    echo "Running APG mode = ${mode}"
    echo "========================================"

    python3 apg.py \
        --csv "outputs/prompts/${mode}.csv" \
        --scale 9 \
        --eta 0 \
        --momentum -0.5 \
        --norm-threshold 0

    python3 evaluate.py --pred "${mode}"

    echo "Finished APG mode=${mode}"
done

echo "Experiment 4 completed."
