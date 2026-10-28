import random
import torch

from pathlib import Path
from statistics import median
from miditok import REMI
from symusic import Score, Tempo
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from tokenization import read_beats
from bonus_tokenization import encode_track

CHECKPOINT_PATH = Path("outputs/bonus/checkpoints/best.pt")
TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
PROMPT_DIR = Path("data/POP909/prompts/bonus")
OUTPUT_DIR = Path("outputs/bonus/generated")

SEED = 42
TOP_K = 50
TEMPERATURE = 1.0
BARS = 16
MAX_LENGTH = 2048

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = T5Config.from_dict(checkpoint["model_config"])
    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"Checkpoint: {checkpoint_path} | Device: {device}")
    return model

def special_ids(tokenizer):
    vocab = tokenizer.vocab
    return vocab["Bar_None"], vocab["PAD_None"], vocab["EOS_None"]

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

def crop_bars(ids, bar_id, start):
    starts = [i for i, token in enumerate(ids) if token == bar_id]
    end = starts[start + BARS] if len(starts) > start + BARS else len(ids)
    return ids[starts[start]:end]

def tokenize_prompt(tokenizer, midi_path):
    beat_path = midi_path.parent / "beat_midi.txt"
    score, target_bpm = corrected_prompt_tempo(midi_path, beat_path)
    ids = encode_track(tokenizer, score, "MELODY")
    return ids, target_bpm

@torch.inference_mode()
def generate(model, prefix, device, pad_id, eos_id):
    x = torch.tensor([prefix[:MAX_LENGTH - 1] + [eos_id]], dtype=torch.long, device=device)

    output = model.generate(
        input_ids=x,
        attention_mask=torch.ones_like(x),
        do_sample=True,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_LENGTH - 1,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )[0].tolist()

    return [token for token in output if token not in {pad_id, eos_id}]

def save_midi(tokenizer, melody_ids, piano_ids, target_bpm, output_path):
    melody_score = tokenizer.decode([melody_ids])
    piano_score = tokenizer.decode([piano_ids])

    melody_track = melody_score.tracks[0]
    piano_track = piano_score.tracks[0]
    melody_track.name = "MELODY"
    piano_track.name = "PIANO"

    score = melody_score
    score.tracks = [melody_track, piano_track]
    score.tempos = [Tempo(0, target_bpm)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(output_path))

def run_bonus(model, tokenizer, bar_id, pad_id, eos_id, device):
    prompts = sorted(PROMPT_DIR.rglob("*.mid"))

    for prompt_path in tqdm(prompts, desc="Bonus"):
        melody_ids, target_bpm = tokenize_prompt(tokenizer, prompt_path)
        print(f"Prompt {prompt_path.stem}: {len(melody_ids)} tokens, {melody_ids.count(bar_id)} Bar tokens")

        piano_ids = generate(model, melody_ids, device, pad_id, eos_id)
        if piano_ids and piano_ids[0] != bar_id:
            piano_ids.insert(0, bar_id)

        midi_path = OUTPUT_DIR / f"{prompt_path.stem}" / f"{prompt_path.stem}.mid"
        save_midi(tokenizer, melody_ids, piano_ids, target_bpm, midi_path)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = REMI(params=str(TOKENIZER_PATH))
    model = load_model(CHECKPOINT_PATH, device)

    bar_id, pad_id, eos_id = special_ids(tokenizer)
    run_bonus(model, tokenizer, bar_id, pad_id, eos_id, device)

    print(f"Done. Outputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
