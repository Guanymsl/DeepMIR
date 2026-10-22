import argparse
import random
import torch
import json

from pathlib import Path
from statistics import median
from miditok import REMI
from symusic import Score, Track, Tempo
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from tokenization import read_beats

CHECKPOINT_PATH = Path("outputs/checkpoints/best.pt")
TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
PROMPT_DIR = Path("data/POP909/prompts/basic-b")
OUTPUT_DIR = Path("outputs/generated")

SEED = 42
SAMPLES = 20
SAMPLING_CONFIGS = {
    "s1": {"top_k": 20, "temperature": 0.8},
    "s2": {"top_k": 50, "temperature": 1.0},
    "s3": {"top_k": 100, "temperature": 1.2},
}
MAX_TOKENS_PER_BAR = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["A", "B"])
    return parser.parse_args()

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["model_config"].copy()
    config_dict["bos_token_id"] = None
    config_dict["eos_token_id"] = None
    config = GPT2Config.from_dict(config_dict)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"Checkpoint: {checkpoint_path} | Device: {device}")
    return model

def bar_ids_and_start(tokenizer):
    bar_id = tokenizer.vocab["Bar_None"]
    return {bar_id}, bar_id

def forbidden_token_ids(tokenizer):
    return [idx for token, idx in tokenizer.vocab.items() if token.split("_", 1)[0] in {"PAD", "BOS", "EOS", "MASK"}]

def merge_piano_tracks(score):
    merged = Track(program=0, is_drum=False, name="PIANO_MERGED")
    for track in score.tracks:
        if not track.is_drum:
            merged.notes.extend(track.notes)
    merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))
    score.tracks = [merged]
    return score

def corrected_prompt_tempo(midi_path, beat_path):
    score = Score(str(midi_path))
    times, _ = read_beats(beat_path)

    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]

    target_bpm = 60.0 / median(intervals)
    original_bpm = score.tempos[0].qpm
    scale = target_bpm / original_bpm

    for track in score.tracks:
        for note in track.notes:
            old_start = note.time
            old_end = note.time + note.duration

            new_start = max(0, round(old_start * scale))
            new_end = round(old_end * scale)

            note.time = new_start
            note.duration = max(1, new_end - new_start)

    score.tempos = [Tempo(0, target_bpm)]

    return score, target_bpm

def tokenize_prompt(tokenizer, midi_path):
    beat_path = midi_path.parent / "beat_midi.txt"
    score, _ = corrected_prompt_tempo(midi_path, beat_path)
    score = merge_piano_tracks(score)
    encoded = tokenizer.encode(score)
    if isinstance(encoded, list):
        encoded = encoded[0]
    return list(encoded.ids)

def sample_next(logits, temperature, top_k, forbidden_ids):
    logits = logits.float().clone() / temperature
    if forbidden_ids:
        logits[forbidden_ids] = -float("inf")
    k = min(top_k, logits.numel())
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    chosen = torch.multinomial(probs, 1)
    return int(indices[chosen].item())

def save_tokens(ids, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f)

@torch.inference_mode()
def generate(model, prefix, bar_ids, config, device, target_bars=32, forbidden_ids=(), max_tokens_per_bar=None):
    ids = list(prefix)
    bars = sum(token in bar_ids for token in ids)

    tokens_in_bar = 0
    for token in reversed(ids):
        if token in bar_ids:
            break
        tokens_in_bar += 1

    while bars <= target_bars:
        if max_tokens_per_bar is not None and tokens_in_bar >= max_tokens_per_bar:
            ids.append(next(iter(bar_ids)))
            bars += 1
            tokens_in_bar = 0
            if bars > target_bars:
                ids.pop()
            continue

        context = torch.tensor(
            [ids[-model.config.n_positions:]],
            dtype=torch.long,
            device=device
        )

        next_logits = model(input_ids=context, use_cache=False).logits[0, -1]
        next_id = sample_next(
            next_logits,
            config["temperature"],
            config["top_k"],
            forbidden_ids
        )

        if next_id in bar_ids:
            bars += 1
            if bars > target_bars:
                break
            tokens_in_bar = 0
        else:
            tokens_in_bar += 1

        ids.append(next_id)

    return ids

def save_midi(tokenizer, ids, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score = tokenizer.decode([ids])
    score.dump_midi(str(output_path))

def run_task_a(model, tokenizer, bar_ids, start_bar_id, forbidden, device):
    for config_name, config in SAMPLING_CONFIGS.items():
        folder = OUTPUT_DIR / "A" / config_name

        for i in tqdm(range(1, SAMPLES + 1), desc=f"Task A {config_name}"):
            ids = generate(model, [start_bar_id], bar_ids, config, device, forbidden_ids=forbidden,  max_tokens_per_bar=MAX_TOKENS_PER_BAR)
            midi_path = folder / f"{i:03d}" / f"{i:03d}.mid"
            save_midi(tokenizer, ids, midi_path)
            save_tokens(ids, midi_path.with_suffix(".json"))

def run_task_b(model, tokenizer, bar_ids, forbidden, device):
    prompts = sorted(PROMPT_DIR.rglob("*.mid"))

    for prompt_path in prompts:
        prefix = tokenize_prompt(tokenizer, prompt_path)
        prompt_bars = sum(token in bar_ids for token in prefix)
        print(f"Prompt {prompt_path.stem}: {len(prefix)} tokens, {prompt_bars} Bar tokens")
        if prompt_bars != 8:
            print(f'WARNING: Expected 8 prompt bars, found {prompt_bars}: {prompt_path}')

        for config_name, config in SAMPLING_CONFIGS.items():
            ids = generate(model, prefix, bar_ids, config, device, forbidden_ids=forbidden)
            midi_path = OUTPUT_DIR / "B" / config_name / f"{prompt_path.stem}" / f"{prompt_path.stem}.mid"
            save_midi(tokenizer, ids, midi_path)
            save_tokens(ids, midi_path.with_suffix(".json"))

def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = REMI(params=str(TOKENIZER_PATH))
    model = load_model(CHECKPOINT_PATH, device)

    bar_ids, start_bar_id = bar_ids_and_start(tokenizer)
    forbidden = forbidden_token_ids(tokenizer)

    if args.task == "A":
        run_task_a(model, tokenizer, bar_ids, start_bar_id, forbidden, device)
    else:
        run_task_b(model, tokenizer, bar_ids, forbidden, device)

    print(f"Done. Outputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
