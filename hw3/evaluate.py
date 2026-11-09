import argparse
import csv
import json
import subprocess
import sys
import torch
import laion_clap

from pathlib import Path

CAPTIONS_PATH = Path("outputs/av.csv")
GT_DIR = Path("audio_wav")
PRED_DIR = Path("outputs/generated")
OUTPUT_DIR = Path("outputs/evaluation")

METRICS = ("fd-vgg", "kl-passt", "clap", "mqascore")

BATCH_SIZE = 8
SECONDS = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--paired-gt', type=Path)
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=[*METRICS, 'all'],
        required=True,
    )
    return parser.parse_args()

def run(cmd):
    print('Running:', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def main():
    args = parse_args()

    script = Path("av-benchmark/evaluate.py")

    pred = PRED_DIR / args.pred
    output = OUTPUT_DIR / args.pred
    metrics = set(METRICS) if 'all' in args.metrics else set(args.metrics)

    output.mkdir(parents=True, exist_ok=True)
    result = {}

    if 'fd-vgg' in metrics:
        gt_cache = output / 'musiccaps_cache'
        pred_cache = output / 'unpaired_cache'

        cmd = [
            sys.executable,
            str(script),
            '--gt_audio', str(GT_DIR),
            '--pred_audio', str(pred),
            '--gt_cache', str(gt_cache),
            '--pred_cache', str(pred_cache),
            '--audio_length', str(SECONDS),
            '--gt_batch_size', '1',
            '--pred_batch_size', str(BATCH_SIZE),
            '--skip_video_related',
            '--skip_clap',
            '--unpaired',
        ]
        run(cmd)

        with (pred_cache / 'output_metrics.json').open() as f:
            data = json.load(f)

        result['FD-VGG'] = data['FD-VGG']

    if 'clap' in metrics:
        checkpoint = Path("av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt")

        with CAPTIONS_PATH.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel='HTSAT-base',
        )
        model.load_ckpt(str(checkpoint))
        model.eval()

        scores = {}

        with torch.inference_mode():
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                texts = [row['caption'] for row in batch]
                files = [
                    next(
                        p for p in pred.iterdir()
                        if p.stem == str(row['name'])
                        and p.suffix.lower() in ('.wav', '.flac', '.mp3')
                    )
                    for row in batch
                ]

                text_emb = model.get_text_embedding(
                    texts, use_tensor=True
                )
                audio_emb = model.get_audio_embedding_from_filelist(
                    x=[str(p) for p in files],
                    use_tensor=True,
                )

                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
                audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
                values = (text_emb * audio_emb).sum(dim=-1)

                for row, score in zip(batch, values):
                    scores[str(row['name'])] = float(score.item())

        result['LAION-CLAP-Score'] = sum(scores.values()) / len(scores)

        (output / 'clap.json').write_text(
            json.dumps(scores, indent=2) + '\n',
            encoding='utf-8',
        )

    if 'kl-passt' in metrics:
        gt_cache = output / 'paired_gt_cache'
        pred_cache = output / 'paired_pred_cache'

        cmd = [
            sys.executable,
            str(script),
            '--gt_audio', str(args.paired_gt),
            '--pred_audio', str(pred),
            '--gt_cache', str(gt_cache),
            '--pred_cache', str(pred_cache),
            '--audio_length', str(SECONDS),
            '--gt_batch_size', '1',
            '--pred_batch_size', str(BATCH_SIZE),
            '--skip_video_related',
            '--skip_clap',
        ]
        run(cmd)

        with (pred_cache / 'output_metrics.json').open() as f:
            data = json.load(f)

        result['KL-PASST-softmax'] = data['KL-PASST-softmax']

    if 'mqascore' in metrics:
        mqa_output = output / 'mqascore'
        mqa_output.mkdir(parents=True, exist_ok=True)

        captions = mqa_output / 'captions.jsonl'
        tags = mqa_output / 'tags.jsonl'
        scores_path = mqa_output / 'scores.jsonl'

        with CAPTIONS_PATH.open(newline='', encoding='utf-8') as f:
            caption_map = {
                str(row['name']): row['caption']
                for row in csv.DictReader(f)
            }

        with captions.open('w', encoding='utf-8') as f:
            for sid in sorted(caption_map, key=int):
                audio = pred / f'{sid}.wav'
                item = {
                    'id': sid,
                    'audio': str(audio.resolve()),
                    'text': caption_map[sid],
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        if not tags.exists():
            cmd = [
                'conda', 'run', '--no-capture-output',
                '-n', 'mqascore_tags',
                'python', 'MQAScore/scripts/extract_tags.py',
                '--model', 'Qwen/Qwen3-1.7B',
                '--input', str(captions),
                '--output', str(tags),
                '--max-model-len', '4096',
                '--max-new-tokens', '1024',
                '--gpu-memory-utilization', '0.85',
            ]
            run(cmd)

        cmd = [
            'conda', 'run', '--no-capture-output',
            '-n', 'mqascore_af2',
            'python', 'MQAScore/scripts/score.py',
            '--model', 'af2',
            '--input', str(tags),
            '--output', str(scores_path),
            '--protocol', 'pertag',
            '--fix-seconds', str(SECONDS),
        ]
        run(cmd)

        with scores_path.open(encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        result['MQAScore'] = sum(float(item['mean_score']) for item in data) / len(data)
        print(f'MQAScore : 'f"{result['MQAScore']:.6f}")

    path = output / 'metrics.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(result, indent=2))
    print('Saved:', path)

if __name__ == '__main__':
    main()
