import json
import random
import itertools
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from miditok import REMI

from MusDr.musdr.side_utils import (
    get_pitch_histogram,
    compute_histogram_entropy,
    get_onset_xor_distance,
)

TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
GENERATED_DIR = Path("outputs/generated/A")
OUTPUT_DIR = Path("outputs/evaluation/")

def get_token_ids(tokenizer):
    vocab = tokenizer.vocab

    bar_id = vocab["Bar_None"]

    pos_keys = [k for k in vocab if k.startswith("Position_")]
    pos_keys.sort(key=lambda k: int(k.split("_", 1)[1]))
    pos_ids = [vocab[k] for k in pos_keys]

    pitch_keys = [k for k in vocab if k.startswith("Pitch_")]
    pitch_keys.sort(key=lambda k: int(k.split("_", 1)[1]))
    pitch_ids = [vocab[k] for k in pitch_keys]

    return bar_id, pos_ids, pitch_ids

def split_bars(ids, bar_id):
    positions = [i for i, token in enumerate(ids) if token == bar_id]

    if not positions:
        return []

    bars = []
    for i, start in enumerate(positions):
        end = (
            positions[i + 1]
            if i + 1 < len(positions)
            else len(ids)
        )
        bars.append(ids[start:end])

    return bars

def compute_pitch_entropy(ids, window_size, bar_id, pitch_ids):
    bars = split_bars(ids, bar_id)

    if not bars:
        return np.nan

    window_size = min(window_size, len(bars))
    values = []

    for start in range(len(bars) - window_size + 1):
        segment = [
            token
            for bar in bars[start:start + window_size]
            for token in bar
        ]

        hist = get_pitch_histogram(
            segment,
            pitch_evs=pitch_ids,
        )

        if hist is None:
            continue

        entropy = compute_histogram_entropy(hist)

        if np.isfinite(entropy):
            values.append(float(entropy))

    return float(np.mean(values)) if values else np.nan

def compute_groove_similarity(
    ids,
    bar_id,
    pos_ids,
    pitch_ids,
    max_pairs=1000,
):
    bars = split_bars(ids, bar_id)
    pairs = list(itertools.combinations(range(len(bars)), 2))

    if len(pairs) > max_pairs:
        pairs = random.Random(42).sample(pairs, max_pairs)

    values = []
    for i, j in pairs:
        distance = get_onset_xor_distance(
            bars[i],
            bars[j],
            bar_id,
            pos_ids,
            pitch_evs=pitch_ids,
        )

        similarity = 1.0 - distance
        if np.isfinite(similarity):
            values.append(float(similarity))

    return float(np.mean(values)) if values else np.nan

def evaluate_file(
    json_path,
    config,
    bar_id,
    pos_ids,
    pitch_ids,
):
    with json_path.open(encoding="utf-8") as f:
        ids = json.load(f)

    bars = split_bars(ids, bar_id)

    return {
        "config": config,
        "song_name": json_path.stem,
        "file": str(json_path),
        "bars": len(bars),
        "H1": compute_pitch_entropy(
            ids, 1, bar_id, pitch_ids
        ),
        "H4": compute_pitch_entropy(
            ids, 4, bar_id, pitch_ids
        ),
        "GS": compute_groove_similarity(
            ids, bar_id, pos_ids, pitch_ids
        ),
    }

def main():
    tokenizer = REMI(params=str(TOKENIZER_PATH))
    bar_id, pos_ids, pitch_ids = get_token_ids(tokenizer)

    paths = []
    for path in sorted((GENERATED_DIR).glob("*/*/*.json")):
        paths.append(("A", path.parent.parent.name, path))

    print(f"Found {len(paths)} generated JSON files")

    results = []
    for task, config, path in tqdm(paths, desc="Evaluating"):
        result = evaluate_file(
            path,
            task,
            config,
            bar_id,
            pos_ids,
            pitch_ids,
        )

        if result is not None:
            results.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    summary = (
        df.groupby(["task", "config"], as_index=False)
        .agg(
            samples=("song_name", "count"),
            mean_H1=("H1", "mean"),
            mean_H4=("H4", "mean"),
            mean_GS=("GS", "mean"),
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    print("\nEvaluation Summary:")
    print(summary.to_string(index=False, float_format="%.4f"))

    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
