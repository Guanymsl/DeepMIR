import json
import math
import random
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from miditok import REMI
from transformers import T5Config, T5ForConditionalGeneration

TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
CHECKPOINT_DIR = Path("outputs/bonus/checkpoints")

BARS = 16
MAX_LENGTH = 2048
SEED = 42

SAMPLES_PER_SONG = 4
VAL_RATIO = 0.1
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1
BATCH_SIZE = 4
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

def special_ids(tokenizer):
    vocab = tokenizer.vocab
    return vocab["Bar_None"], vocab["PAD_None"], vocab["EOS_None"]

def crop_bars(ids, bar_id, start, bars=BARS):
    starts = [i for i, token in enumerate(ids) if token == bar_id]
    if len(starts) < start + bars:
        return None
    end = starts[start + bars] if len(starts) > start + bars else len(ids)
    return ids[starts[start]:end]

def load_songs(tokenizer, bar_id):
    with TOKENS_PATH.open(encoding="utf-8") as f:
        records = json.load(f)
    songs = [(r["melody_ids"], r["piano_ids"]) for r in records
             if min(r["melody_ids"].count(bar_id), r["piano_ids"].count(bar_id)) >= BARS]

    return songs

class AccompanimentDataset(Dataset):
    def __init__(self, songs, bar_id, pad_id, eos_id, samples_per_song, seed=None):
        self.songs = songs
        self.bar_id = bar_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.samples_per_song = samples_per_song
        self.seed = seed

    def __len__(self):
        return len(self.songs) * self.samples_per_song

    def __getitem__(self, index):
        song_idx = index // self.samples_per_song
        sample_idx = index % self.samples_per_song
        melody, piano = self.songs[song_idx]
        available = min(melody.count(self.bar_id), piano.count(self.bar_id)) - BARS
        if self.seed is None:
            start = random.randint(0, available)
        else:
            start = random.Random(self.seed + song_idx * 1000003 + sample_idx).randint(0, available)
        x = crop_bars(melody, self.bar_id, start)[:MAX_LENGTH - 1] + [self.eos_id]
        y = crop_bars(piano, self.bar_id, start)[:MAX_LENGTH - 1] + [self.eos_id]
        return x, y

def collate(batch, pad_id):
    max_x = max(len(x) for x, _ in batch)
    max_y = max(len(y) for _, y in batch)
    input_ids = torch.tensor([x + [pad_id] * (max_x - len(x)) for x, _ in batch])
    attention_mask = (input_ids != pad_id).long()
    labels = torch.tensor([y + [-100] * (max_y - len(y)) for _, y in batch])
    return input_ids, attention_mask, labels

def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, mask, y in tqdm(loader, desc="Train" if train else "Valid", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            loss = model(input_ids=x, attention_mask=mask, labels=y).loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        tokens = (y != -100).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / max(total_tokens, 1)

def train(tokenizer, device):
    bar_id, pad_id, eos_id = special_ids(tokenizer)
    songs = load_songs(tokenizer, bar_id)
    random.Random(SEED).shuffle(songs)
    n_val = max(1, min(len(songs) - 1, round(len(songs) * VAL_RATIO)))
    val_songs, train_songs = songs[:n_val], songs[n_val:]
    train_data = AccompanimentDataset(train_songs, bar_id, pad_id, eos_id, SAMPLES_PER_SONG)
    val_data = AccompanimentDataset(val_songs, bar_id, pad_id, eos_id, 1, seed=SEED)
    collate_fn = lambda batch: collate(batch, pad_id)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    config = T5Config(
        vocab_size=len(tokenizer), d_model=D_MODEL, d_ff=D_MODEL * 4,
        num_heads=N_HEADS, num_layers=N_LAYERS, num_decoder_layers=N_LAYERS,
        dropout_rate=DROPOUT, pad_token_id=pad_id,
        decoder_start_token_id=pad_id, eos_token_id=eos_id,
    )
    model = T5ForConditionalGeneration(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Songs: {len(train_songs)} train / {len(val_songs)} val")

    best_val = math.inf
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, optimizer, device, False)
        print(f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        checkpoint = {"model_state_dict": model.state_dict(), "model_config": config.to_dict(),
                      "epoch": epoch, "val_loss": val_loss}
        torch.save(checkpoint, CHECKPOINT_DIR / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
        with (CHECKPOINT_DIR / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = REMI(params=str(TOKENIZER_PATH))
    train(tokenizer, device)

    print("Done.")

if __name__ == "__main__":
    main()
