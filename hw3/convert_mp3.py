"""Convert baseline WAV files to submission MP3 files using ffmpeg."""
import argparse, subprocess
from pathlib import Path
from hw3.common import read_captions

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--captions', default='test.csv')
    p.add_argument('--input', default='outputs/baseline')
    p.add_argument('--output', default='submission/student_id/generated_samples')
    p.add_argument('--bitrate', default='192k')
    args = p.parse_args()
    rows = read_captions(args.captions)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    missing = []
    for sample_id, _ in rows:
        src = Path(args.input) / f'{sample_id}.wav'
        if not src.exists(): missing.append(sample_id); continue
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-codec:a','libmp3lame','-b:a',args.bitrate,str(out / f'{sample_id}.mp3')], check=True)
    if missing: raise SystemExit(f'Missing WAV IDs: {missing}')
    print(f'Converted {len(rows)} files to {out}')

if __name__ == '__main__': main()
