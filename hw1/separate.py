import subprocess
import shutil

import pandas as pd

from pathlib import Path
from tqdm import tqdm

MODEL = "htdemucs"

def main():
    dataset_dir = Path("data/dataset_B")
    output_dir = Path("data/dataset_C")
    temp_dir = Path("outputs/demucs")

    df = pd.read_csv(dataset_dir / "manifest.csv")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Separating vocals"):
        audio_path = dataset_dir / row["audio_path"]
        output_path = output_dir / row["audio_path"]

        if output_path.exists():
            continue

        subprocess.run([
            "python", "-m", "demucs",
            "-n", MODEL,
            "--two-stems", "vocals",
            "--out", str(temp_dir),
            str(audio_path)
        ], check=True)

        vocal_path = temp_dir / MODEL / audio_path.stem / "vocals.wav"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(vocal_path, output_path)

        shutil.rmtree(temp_dir / MODEL / audio_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset_dir / "manifest.csv", output_dir / "manifest.csv")

if __name__ == "__main__":
    main()
