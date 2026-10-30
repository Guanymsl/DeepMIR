import argparse
import json
import subprocess
import sys

from pathlib import Path

from hw3.common import read_captions, audio_files
from hw3.clap_utils import load_clap, embed_text, embed_audio, DEFAULT_CKPT

CAPTIONS_PATH = Path("test.csv")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['clap','official'], default='clap')
    parser.add_argument('--pred', default='outputs/baseline')
    parser.add_argument('--gt', help='reference WAV/FLAC folder; paired matching filenames needed for KL')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
    parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', default='outputs/evaluation/clap_music.json')
    parser.add_argument('--av-benchmark')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--unpaired', action='store_true')
    return parser.parse_args()

def clap_score(args):
    rows = read_captions(CAPTIONS_PATH)

    by_id = {p.stem: p for p in audio_files(args.pred)}

    model = load_clap(args.checkpoint, args.device)
    te = embed_text(model, [caption for _, caption in rows])
    ae = embed_audio(model, [by_id[sid] for sid, _ in rows], args.batch_size)
    scores = (te * ae).sum(axis=1)

    result = {'CLAP_music_mean_cosine': float(scores.mean()), 'CLAP_music_std_cosine': float(scores.std()),
              'checkpoint': str(args.checkpoint), 'num_samples': len(rows),
              'per_sample': {sid: float(score) for (sid, _), score in zip(rows, scores)}}

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    print(f'CLAP (music checkpoint): {scores.mean():.6f}  ({len(rows)} samples)')
    print('Saved:', out)

def official(args):
    repo = Path(args.av_benchmark).expanduser().resolve()
    script = repo / 'evaluate.py'

    cmd = [sys.executable, str(script), '--gt_audio', str(Path(args.gt).resolve()),
           '--pred_audio', str(Path(args.pred).resolve()), '--audio_length', str(args.seconds),
           '--skip_video_related', '--skip_clap']
    if args.unpaired: cmd.append('--unpaired')
    print('Running official av-benchmark:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)

def main():
    args = parse_args()

    if args.mode == 'clap':
        clap_score(args)
    else:
        official(args)

if __name__ == '__main__':
    main()
