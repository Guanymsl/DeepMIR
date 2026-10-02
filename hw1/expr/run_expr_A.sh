#!/bin/bash

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:${DYLD_LIBRARY_PATH:-}"

PYTHON="./venv/bin/python"

run() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
    $PYTHON "$@"
}

run expr/A/A_mean.py

run expr/A/A_mean_std.py

run expr/A/A_mean_std_aug.py

for layer in 0 3 6 9 12 15 18 21 24; do
    run expr/A/A_mean_std_aug_layer.py \
        --layer "$layer"
done
