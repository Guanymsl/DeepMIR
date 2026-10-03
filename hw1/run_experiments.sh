#!/bin/bash

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:${DYLD_LIBRARY_PATH:-}"

PYTHON="./venv/bin/python"
TRAIN="./train.py"

run() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
    $PYTHON $TRAIN "$@"
}

run --dataset A --pooling mean --layer last
run --dataset A --pooling meanstd --layer last
run --dataset A --pooling mean --layer last --augment
run --dataset A --pooling meanstd --layer last --augment

for layer in 0 3 6 9 12 15 18 21 24; do
    run --dataset A --pooling meanstd --layer "$layer"
done

run --dataset A --pooling meanstd --layer last --acoustic

run --dataset B --pooling mean --layer last
run --dataset B --pooling meanstd --layer last
run --dataset B --pooling mean --layer last --augment
run --dataset B --pooling meanstd --layer last --augment

for layer in 0 3 6 9 12 15 18 21 24; do
    run --dataset B --pooling mean --layer "$layer" --augment
done

run --dataset B --pooling mean --layer 6 --augment --vocal
