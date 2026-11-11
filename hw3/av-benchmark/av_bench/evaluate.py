import logging
from pathlib import Path
from typing import Dict

import torch

from av_bench.metrics.fad import compute_fd
from av_bench.metrics.kl import compute_kl
from av_bench.utils import unroll_dict, unroll_paired_dict

log = logging.getLogger()
device = 'cuda'

@torch.inference_mode()
def evaluate(gt_audio_cache: Path,
             pred_audio_cache: Path,
             *,
             is_paired: bool = True,
             num_samples: int = 1,
             skip_video_related: bool = False,
             skip_clap: bool = False) -> Dict[str, float]:

    gt_audio_cache = gt_audio_cache.expanduser()
    pred_audio_cache = pred_audio_cache.expanduser()

    gt_vggish_features = torch.load(gt_audio_cache / 'vggish_features.pth', weights_only=True)
    pred_vggish_features = torch.load(pred_audio_cache / 'vggish_features.pth', weights_only=True)

    gt_passt_logits = torch.load(gt_audio_cache / 'passt_logits.pth', weights_only=True)
    pred_passt_logits = torch.load(pred_audio_cache / 'passt_logits.pth', weights_only=True)

    if is_paired:
        gt_passt_logits, pred_passt_logits, unpaired_passt_keys = unroll_paired_dict(
            gt_passt_logits, pred_passt_logits)
        log.info(f'Unpaired PASST logits keys: {unpaired_passt_keys}')

    else:
        gt_passt_logits = unroll_dict(gt_passt_logits)
        pred_passt_logits = unroll_dict(pred_passt_logits)

    gt_vggish_features = unroll_dict(gt_vggish_features, cat=True)
    pred_vggish_features = unroll_dict(pred_vggish_features, cat=True)

    output_metrics = {}

    fd_vgg = compute_fd(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['FD-VGG'] = fd_vgg

    if is_paired:
        kl_metrics = compute_kl([pred_passt_logits], gt_passt_logits)
        output_metrics['KL-PASST-softmax'] = kl_metrics['kl_softmax']

    return output_metrics
