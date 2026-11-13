import subprocess
import argparse

from pathlib import Path

from common import read_captions

CAPTIONS_PATH = Path('test.csv')
INPUT_DIR = Path('outputs/generated')
OUTPUT_DIR = Path('outputs/converted')
BITRATE = '192k'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = INPUT_DIR / args.pred
    output = OUTPUT_DIR / args.pred
    output.mkdir(parents=True, exist_ok=True)

    rows = read_captions(CAPTIONS_PATH)

    for sample_id, _ in rows:
        src = input_dir / f'{sample_id}.wav'
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
            str(output / f'{sample_id}.mp3')],
            check=True,
        )

    print(f'Converted {len(rows)} files to {output}')

if __name__ == '__main__':
    main()
