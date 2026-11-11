import argparse
import csv
import json
import subprocess
import sys
import shutil

from pathlib import Path

CAPTIONS_PATH = Path('av.csv')
GT_DIR = Path('audio_wav')
GT_PATH = Path('outputs/retrieval.json')

PRED_DIR = Path('outputs/generated')
OUTPUT_DIR = Path('outputs/evaluation')

METRICS = ('fd-vgg', 'kl-passt', 'clap', 'mqascore')

BATCH_SIZE = 8
SECONDS = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=[*METRICS, 'all'],
        default=['all'],
    )
    return parser.parse_args()

def run(cmd):
    print('Running:', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def main():
    args = parse_args()

    script = Path('av-benchmark/evaluate.py')

    pred = PRED_DIR / args.pred
    output = OUTPUT_DIR / args.pred
    metrics = set(METRICS) if 'all' in args.metrics else set(args.metrics)

    output.mkdir(parents=True, exist_ok=True)
    result = {}

    if 'fd-vgg' in metrics:
        gt_cache = output / 'unpaired_gt_cache'
        pred_cache = output / 'unpaired_pred_cache'

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
        print(f"FD-VGG: {result['FD-VGG']:.6f}")

    if 'clap' in metrics:
        clap_cache = output / 'clap_cache'
        cmd = [
            sys.executable,
            str(script),
            '--clap_only',
            '--captions', str(CAPTIONS_PATH),
            '--pred_audio', str(pred),
            '--pred_cache', str(clap_cache),
            '--audio_length', str(SECONDS),
            '--pred_batch_size', str(BATCH_SIZE),
        ]
        run(cmd)
        with (clap_cache / 'output_metrics.json').open() as f:
            result['CLAP'] = json.load(f)['LAION-CLAP-Score']
        print(f"CLAP: {result['CLAP']:.6f}")

    if 'kl-passt' in metrics:
        retrieval_path = Path('outputs/retrieval.json')
        reference_dir = output / 'retrieved_reference'
        reference_dir.mkdir(parents=True, exist_ok=True)

        with retrieval_path.open(encoding='utf-8') as f:
            retrieval = json.load(f)

        for sid, musiccaps_id in retrieval.items():
            source = GT_DIR / f'{musiccaps_id}.wav'
            destination = reference_dir / f'{sid}.wav'
            shutil.copyfile(source, destination)

        gt_cache = output / 'paired_gt_cache'
        pred_cache = output / 'paired_pred_cache'

        for cache in (gt_cache, pred_cache):
            if cache.exists():
                shutil.rmtree(cache)

        cmd = [
            sys.executable,
            str(script),
            '--gt_audio', str(reference_dir),
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

        result['KL-passt'] = data['KL-PASST-softmax']
        print(f"KL-passt: {result['KL-passt']:.6f}")

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
        print(f"MQAScore : {result['MQAScore']:.6f}")

    path = output / 'metrics.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(result, indent=2))
    print('Saved:', path)

if __name__ == '__main__':
    main()
