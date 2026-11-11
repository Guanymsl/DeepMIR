import csv
import subprocess

from pathlib import Path
from tqdm import tqdm

CAPTIONS_PATH = Path("test.csv")
OUTPUT_PATH = Path("av.csv")

MP3_DIR = Path("audio_mp3")
WAV_DIR = Path("audio_wav")

def prepare_csv():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CAPTIONS_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "caption"])
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "name": row["id"],
                "caption": row["caption"],
            })

    print(f"Created {OUTPUT_PATH}")

def convert_audio(mp3_path):
    wav_path = WAV_DIR / f"{mp3_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-y",
        "-i", str(mp3_path),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    return mp3_path.name

def prepare_audio():
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in MP3_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp3"
    )

    print(f"Found {len(files)} MP3 files")

    for file in tqdm(files, desc="MP3 → WAV"):
        convert_audio(file)

    print(f"Created {WAV_DIR}")

def main():
    prepare_csv()
    prepare_audio()

if __name__ == "__main__":
    main()
