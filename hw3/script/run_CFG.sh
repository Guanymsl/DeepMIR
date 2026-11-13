#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for scale in 1 3 5 7 9 11 13; do
    echo "================================"
    echo "Running CFG scale = ${scale}"
    echo "================================"

    python3 baseline.py --scale "${scale}.0"
    python3 evaluate.py --pred "baseline${scale}"

    echo "Finished CFG scale = ${scale}"
done

echo "All CFG ablation experiments completed!"
