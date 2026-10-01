#!/bin/bash

set -e

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:${DYLD_LIBRARY_PATH:-}"

PYTHON="./venv/bin/python"
TRAIN="train.py"

run() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
    $PYTHON $TRAIN "$@"
}

for dataset in A B; do
    run \
        --dataset "$dataset" \
        --mert 95M \
        --layer -1

    run \
        --dataset "$dataset"  \
        --mert 330M \
        --layer -1

    run \
        --dataset "$dataset" \
        --mert 330M \
        --layer -1 \
        --augment

    for layer in 3 6 9 12 15 18 21; do
        run \
            --dataset "$dataset" \
            --mert 330M \
            --layer "$layer" \
            --augment
    done
done
