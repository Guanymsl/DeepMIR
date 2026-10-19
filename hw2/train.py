import json
import math
import random
import torch

from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from miditok import REMI
from transformers import GPT2Config, GPT2LMHeadModel

class RandomWindowDataset(Dataset):
    def __init__(self, songs, context_length, samples_per_song, seed=42):
        self.songs = [s for s in songs if len(s) >= context_length + 1]
        self.context_length = context_length
        self.samples_per_song = samples_per_song
        self.seed = seed

    def __len__(self):
        return len(self.songs) * self.samples_per_song

    def __getitem__(self, index):
        song_idx = index // self.samples_per_song
        sample_idx = index % self.samples_per_song
        seq = self.songs[song_idx]
        if self.seed is None:
            start = random.randint(0, len(seq) - self.context_length - 1)
        else:
            rng = random.Random(self.seed + song_idx * 1000003 + sample_idx)
            start = rng.randint(0, len(seq) - self.context_length - 1)
        window = seq[start:start + self.context_length + 1]
        return torch.tensor(window[:-1], dtype=torch.long), torch.tensor(window[1:], dtype=torch.long)

def load(path):
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    songs = []
    for record in records:
        sequences = record.get("sequences", [])
        ids = sequences[0]["ids"]
        songs.append(ids)
    return songs

def run_epoch(model, loader, optimizer, device, train, grad_clip):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in tqdm(loader, desc="Train" if train else "Valid", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(input_ids=x, use_cache=False).logits
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        tokens = y.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens

TOKENS_PATH = Path("outputs/tokens/tokens.json")
TOKENIZER_PATH = Path("outputs/tokens/tokenizer.json")
OUTPUT_DIR = Path("outputs/checkpoints")

CONTEXT_LENGTH = 1024
SAMPLES_PER_SONG = 16
VAL_SAMPLES_PER_SONG = 4
VAL_RATIO = 0.1

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = REMI(params=str(TOKENIZER_PATH))
    vocab_size = len(tokenizer)

    songs = load(TOKENS_PATH)
    rng = random.Random(SEED)
    rng.shuffle(songs)
    split = max(1, min(len(songs) - 1, round(len(songs) * VAL_RATIO)))
    val_songs, train_songs = songs[:split], songs[split:]
    train_data = RandomWindowDataset(train_songs, CONTEXT_LENGTH, SAMPLES_PER_SONG, seed=None)
    val_data = RandomWindowDataset(val_songs, CONTEXT_LENGTH, VAL_SAMPLES_PER_SONG, seed=SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=D_MODEL,
        n_head=N_HEADS,
        n_layer=N_LAYERS,
        resid_pdrop=DROPOUT,
        embd_pdrop=DROPOUT,
        attn_pdrop=DROPOUT,
        use_cache=False
    )
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Songs: {len(train_songs)} train / {len(val_songs)} val")

    best_val = math.inf
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True, GRAD_CLIP)
        val_loss = run_epoch(model, val_loader, optimizer, device, False, GRAD_CLIP)
        print(f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={math.exp(min(val_loss, 50)):.2f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "model_config": config.to_dict(), "epoch": epoch, "val_loss": val_loss,
                      "tokenizer_path": str(TOKENIZER_PATH)}

        torch.save(checkpoint, OUTPUT_DIR / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, OUTPUT_DIR / "best.pt")

        with (OUTPUT_DIR / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best validation loss: {best_val:.4f} | Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
