import json
import logging
from pathlib import Path

import torch
from colorlog import ColoredFormatter

from av_bench.args import get_eval_parser
from av_bench.evaluate import evaluate
from av_bench.extract import extract

log = logging.getLogger()
device = 'cuda'

LOGFORMAT = '[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s'


def setup_eval_logging(log_level: int = logging.INFO):
    stream = logging.StreamHandler()
    stream.setFormatter(ColoredFormatter(LOGFORMAT))
    logging.basicConfig(level=log_level, handlers=[stream], force=True)
    logging.getLogger('httpx').setLevel(logging.WARNING)


setup_eval_logging()


@torch.inference_mode()
def main(args):
    if args.clap_only:
        if args.captions is None or args.pred_audio is None:
            raise ValueError('--clap_only requires --captions and --pred_audio')
        from av_bench.clap import evaluate_clap

        output = args.pred_cache or args.pred_audio / 'cache'
        output = output.expanduser()
        output.mkdir(parents=True, exist_ok=True)
        score, scores = evaluate_clap(
            args.pred_audio.expanduser(), args.captions.expanduser(),
            args.audio_length, args.pred_batch_size,
        )
        (output / 'output_metrics.json').write_text(
            json.dumps({'LAION-CLAP-Score': score}, indent=2) + '\n', encoding='utf-8')
        (output / 'clap.json').write_text(
            json.dumps(scores, indent=2) + '\n', encoding='utf-8')
        log.info('LAION-CLAP-Score: %.6f', score)
        return

    gt_audio: Path = args.gt_audio
    gt_cache: Path = args.gt_cache
    pred_audio: Path = args.pred_audio
    pred_cache: Path = args.pred_cache
    audio_length: float = args.audio_length
    num_workers: int = args.num_workers
    gt_batch_size: int = args.gt_batch_size
    pred_batch_size: int = args.pred_batch_size
    unpaired: bool = args.unpaired
    skip_video_related: bool = args.skip_video_related
    skip_clap: bool = args.skip_clap
    recompute_gt_cache: bool = args.recompute_gt_cache
    recompute_pred_cache: bool = args.recompute_pred_cache

    # apply default path
    if gt_cache is None:
        if gt_audio is None:
            raise ValueError('Must specify either gt_audio or gt_cache')
        gt_cache = gt_audio / 'cache'
        log.info(f'No gt cache specified, using default {gt_cache}')

    if pred_cache is None:
        if pred_audio is None:
            raise ValueError('Must specify either pred_audio or pred_cache')
        pred_cache = pred_audio / 'cache'
        log.info(f'No pred cache specified, using default {pred_cache}')

    gt_cache = gt_cache.expanduser()
    pred_cache = pred_cache.expanduser()

    log.info(f'GT cache path: {gt_cache}')
    log.info(f'Pred cache: {pred_cache}')
    log.info(f'Audio length: {audio_length}')
    log.info(f'Unpaired: {unpaired}')

    # extract for GT if needed
    if any(not (gt_cache / f).exists() for f in ('passt_logits.pth', 'vggish_features.pth')) or recompute_gt_cache:
        log.info('Extracting GT cache...')
        if gt_audio is None:
            raise ValueError('Must specify gt_audio to compute gt_cache')
        gt_audio = gt_audio.expanduser()
        log.info(f'GT audio path: {gt_audio}')
        extract(
            audio_path=gt_audio,
            output_path=gt_cache,
            audio_length=audio_length,
            device=device,
            batch_size=gt_batch_size,
            num_workers=num_workers,
            skip_video_related=True,
            skip_clap=True,
        )

    # extract for pred if needed
    if any(not (pred_cache / f).exists() for f in ('passt_logits.pth', 'vggish_features.pth')) or recompute_pred_cache:
        log.info('Extracting pred cache...')
        if pred_audio is None:
            raise ValueError('Must specify pred_audio to compute pred_cache')
        pred_audio = pred_audio.expanduser()
        log.info(f'Pred audio path: {pred_audio}')
        extract(
            audio_path=pred_audio,
            output_path=pred_cache,
            audio_length=audio_length,
            device=device,
            batch_size=pred_batch_size,
            num_workers=num_workers,
            skip_video_related=skip_video_related,
            skip_clap=skip_clap,
        )

    log.info('Starting evaluation...')

    num_samples = 1
    output_metrics = evaluate(gt_audio_cache=gt_cache,
                              pred_audio_cache=pred_cache,
                              num_samples=num_samples,
                              is_paired=not unpaired)

    for k, v in output_metrics.items():
        # pad k to 10 characters
        # pad v to 10 decimal places
        log.info(f'{k:<10}: {v:.10f}')

    # write output metrics to file
    output_metrics_file = pred_cache / 'output_metrics.json'
    with open(output_metrics_file, 'w') as f:
        json.dump(output_metrics, f, indent=4)
    log.info(f'Output metrics written to {output_metrics_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = get_eval_parser().parse_args()
    main(args)
