import csv

from pathlib import Path

def read_captions(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(str(r['id']).strip(), r['caption'].strip()) for r in rows]

def audio_files(root):
    root = Path(root)
    files = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}))
    return files
