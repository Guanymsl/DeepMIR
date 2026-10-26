import torch
from pathlib import Path
from tqdm import tqdm
from miditok import REMI
from symusic import Score, Track
from transformers import T5Config, T5ForConditionalGeneration

TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
CHECKPOINT_DIR = Path("outputs/bonus/checkpoints")
BARS = 16
MAX_LENGTH = 2048
SEED = 42

import random

PROMPT_DIR = Path("data/POP909/prompts/bonus")
OUTPUT_DIR = Path("outputs/bonus/generated")
TOP_K = 50
TEMPERATURE = 1.0

def special_ids(tokenizer):
    vocab = tokenizer.vocab
    return vocab["Bar_None"], vocab["PAD_None"], vocab["EOS_None"]

from tokenization_bonus import encode_track

def crop_bars(ids, bar_id, start, bars=BARS):
    starts = [i for i, token in enumerate(ids) if token == bar_id]
    if len(starts) < start + bars:
        return None
    end = starts[start + bars] if len(starts) > start + bars else len(ids)
    return ids[starts[start]:end]

def save_midi(tokenizer, melody_ids, piano_ids, output_path):
    melody_score = tokenizer.decode([melody_ids])
    piano_score = tokenizer.decode([piano_ids])
    melody_track = melody_score.tracks[0]
    piano_track = piano_score.tracks[0]
    melody_track.name = "MELODY"
    piano_track.name = "PIANO"
    score = melody_score
    score.tracks = [melody_track, piano_track]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(output_path))

def generate(tokenizer, device):
    bar_id, pad_id, eos_id = special_ids(tokenizer)
    checkpoint = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = T5ForConditionalGeneration(T5Config.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    paths = sorted(PROMPT_DIR.rglob("*.mid"))
    if not paths:
        raise FileNotFoundError(f"No MIDI prompts found in {PROMPT_DIR}")

    for path in tqdm(paths, desc="Generating accompaniment"):
        score = Score(str(path))
        melody_ids = encode_track(tokenizer, score, "MELODY")
        if melody_ids is None:
            raise ValueError(f"MELODY track not found: {path}")
        melody_ids = crop_bars(melody_ids, bar_id, 0)
        if melody_ids is None:
            raise ValueError(f"Fewer than {BARS} melody bars: {path}")
        x = torch.tensor([melody_ids[:MAX_LENGTH - 1] + [eos_id]], device=device)
        output = model.generate(
            input_ids=x, attention_mask=torch.ones_like(x),
            do_sample=True, top_k=TOP_K, temperature=TEMPERATURE,
            max_new_tokens=MAX_LENGTH - 1,
        )[0].tolist()
        piano_ids = [token for token in output if token not in {pad_id, eos_id}]
        if piano_ids and piano_ids[0] != bar_id:
            piano_ids.insert(0, bar_id)
        output_path = OUTPUT_DIR / f"{path.stem}.mid"
        save_midi(tokenizer, melody_ids, piano_ids, output_path)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = REMI(params=str(TOKENIZER_PATH))
    generate(tokenizer, device)
    print(f"Done. Outputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
