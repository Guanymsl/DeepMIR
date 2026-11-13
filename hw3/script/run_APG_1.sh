#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for scale in 3 5 7 9 11 13; do
    echo "========================================"
    echo "Running APG scale=${scale}"
    echo "========================================"

    python3 apg.py \
        --scale "$scale" \
        --eta 0 \
        --momentum -0.5 \
        --norm-threshold 0

    python3 evaluate.py  --pred "apg_w${scale}_eta0_m-0.5_n0"

    echo "Finished APG scale=${scale}"
done

echo "Experiment 1 completed."
