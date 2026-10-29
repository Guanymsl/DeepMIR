import argparse
import subprocess

from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("outputs/generated")
OUTPUT_DIR = Path("outputs/rendered")
SOUNDFONT_PATH = Path("soundfonts/FluidR3_GM.sf2")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["A", "B", "O", "U"])
    return parser.parse_args()

def render_midi(midi_path, wav_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "fluidsynth",
        "-ni",
         "-o", "synth.polyphony=1024",
        "-F", str(wav_path),
        "-r", "44100",
        str(SOUNDFONT_PATH),
        str(midi_path)
    ], check=True, stdout=subprocess.DEVNULL)

def main():
    args = parse_args()

    if args.task == "O":
        input_dir = Path("data/POP909/training")
        output_dir = OUTPUT_DIR / args.task
        midi_files = sorted(input_dir.rglob("*.mid"))

        for midi_path in tqdm(midi_files, desc=f"Original"):
            wav_path = output_dir / f"{midi_path.stem}.wav"
            render_midi(midi_path, wav_path)

    elif args.task == "U":
        input_dir = Path("outputs/bonus/generated")
        output_dir = Path("outputs/bonus/rendered")
        midi_files = sorted(input_dir.rglob("*.mid"))

        for midi_path in tqdm(midi_files, desc=f"Bonus"):
            wav_path = output_dir / f"{midi_path.stem}.wav"
            render_midi(midi_path, wav_path)

    else:
        input_dir = INPUT_DIR / args.task
        output_dir = OUTPUT_DIR / args.task
        midi_files = sorted(input_dir.rglob("*.mid"))

        for midi_path in tqdm(midi_files, desc=f"Task {args.task}"):
            config_name = midi_path.relative_to(input_dir).parts[0]
            wav_path = output_dir / config_name / f"{midi_path.stem}.wav"
            render_midi(midi_path, wav_path)

    print(f"Done. Rendered {len(midi_files)} WAV files.")

if __name__ == "__main__":
    main()
