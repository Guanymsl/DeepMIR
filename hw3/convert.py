import subprocess

from pathlib import Path

from common import read_captions

CAPTIONS_PATH = Path("test.csv")
INPUT_DIR = Path("outputs/generated")
OUTPUT_DIR = Path("outputs/converted")
BITRATE = "192k"

def main():
    rows = read_captions(CAPTIONS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_id, _ in rows:
        src = INPUT_DIR / f'{sample_id}.wav'
        subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            '-i',
            str(src),
            '-codec:a',
            'libmp3lame',
            '-b:a',
            BITRATE,
            str(OUTPUT_DIR / f'{sample_id}.mp3')],
            check=True,
        )

    print(f'Converted {len(rows)} files to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
