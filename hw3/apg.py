import argparse
import torch
import soundfile as sf

from pathlib import Path
from tqdm import tqdm
from diffusers import StableAudioPipeline
from contextlib import contextmanager

from common import read_captions

MODEL_PATH = 'stabilityai/stable-audio-open-1.0'
CAPTIONS_PATH = Path('test.csv')

SEED = 42
STEPS = 100
SECONDS = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('test.csv'))
    parser.add_argument('--scale', type=float, default=7.0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=-0.5)
    parser.add_argument('--norm-threshold', type=float, default=0.0)
    return parser.parse_args()

class MomentumBuffer:
    def __init__(self, momentum):
        self.momentum = momentum
        self.running_average = 0

    def update(self, value):
        self.running_average = value + self.momentum * self.running_average

def project(v0, v1):
    dtype = v0.dtype
    v0 = v0.double()
    v1 = v1.double()

    dims = tuple(range(1, v0.ndim))
    v1 = torch.nn.functional.normalize(v1, dim=dims)

    parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    orthogonal = v0 - parallel

    return parallel.to(dtype), orthogonal.to(dtype)

def adaptive_projected_guidance(
    pred_cond,
    pred_uncond,
    guidance_scale,
    momentum_buffer=None,
    eta=1.0,
    norm_threshold=0.0
):
    diff = pred_cond - pred_uncond

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        dims = tuple(range(1, diff.ndim))
        norm = diff.norm(p=2, dim=dims, keepdim=True)
        factor = torch.minimum(
            torch.ones_like(norm),
            norm_threshold / norm.clamp_min(1e-8)
        )
        diff = diff * factor

    parallel, orthogonal = project(diff, pred_cond)
    update = orthogonal + eta * parallel
    
    #print('APG called', flush=True)

    return pred_cond + (guidance_scale - 1) * update

@contextmanager
def apply_apg(pipe, args):
    transformer = pipe.transformer
    scheduler = pipe.scheduler

    original_forward = transformer.forward
    original_step = scheduler.step

    predictions = {}
    momentum_buffer = (
        MomentumBuffer(args.momentum)
        if args.momentum != 0 else None
    )

    def patched_forward(*forward_args, **forward_kwargs):
        output = original_forward(*forward_args, **forward_kwargs)

        uncond, cond = output[0].chunk(2, dim=0)
        predictions['uncond'] = uncond
        predictions['cond'] = cond

        return output

    def patched_step(
        model_output,
        timestep,
        sample,
        generator=None,
        return_dict=True
    ):
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)

        sigma = scheduler.sigmas[scheduler.step_index].to(
            device=sample.device,
            dtype=torch.float32
        )

        sample_fp32 = sample.float()
        uncond = predictions.pop('uncond').float()
        cond = predictions.pop('cond').float()

        pred_uncond = scheduler.precondition_outputs(
            sample_fp32, uncond, sigma
        )
        pred_cond = scheduler.precondition_outputs(
            sample_fp32, cond, sigma
        )

        pred_guided = adaptive_projected_guidance(
            pred_cond,
            pred_uncond,
            args.scale,
            momentum_buffer,
            args.eta,
            args.norm_threshold
        )

        sigma_data = scheduler.config.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)

        c_out = -sigma * sigma_data / (
            sigma ** 2 + sigma_data ** 2
        ).sqrt()

        guided_output = (
            pred_guided - c_skip * sample_fp32
        ) / c_out

        guided_output = guided_output.to(model_output.dtype)
        
        #cfg_pred = pred_uncond + args.scale * (pred_cond - pred_uncond)

        #difference = (pred_guided - cfg_pred).float()

        #print(
        #    'APG vs CFG:',
        #    difference.abs().mean().item(),
        #    flush=True
        #)
        
        #print(
        #    'Scheduler input difference:',
        #    (guided_output.float() - model_output.float()).abs().mean().item(),
        #    flush=True
        #)

        return original_step(
            guided_output,
            timestep,
            sample,
            generator=generator,
            return_dict=return_dict
        )

    transformer.forward = patched_forward
    scheduler.step = patched_step

    try:
        yield
    finally:
        transformer.forward = original_forward
        scheduler.step = original_step

def main():
    args = parse_args()

    if args.csv == Path('test.csv'):
        output_dir = Path(
            f'outputs/generated/apg_w{args.scale:g}_eta{args.eta:g}'
            f'_m{args.momentum:g}_n{args.norm_threshold:g}'
        )
    else:
        output_dir = Path('outputs/generated') / args.csv.stem

    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_captions(args.csv)
    pipe = StableAudioPipeline.from_pretrained(MODEL_PATH, dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    for sample_id, caption in tqdm(rows, desc='Generate'):
        path = output_dir / f'{sample_id}.wav'

        seed = SEED + int(sample_id) if sample_id.isdecimal() else SEED
        gen = torch.Generator(device='cpu').manual_seed(seed)

        with torch.inference_mode(), apply_apg(pipe, args):
            result = pipe(
                prompt=caption,
                negative_prompt='',
                num_inference_steps=STEPS,
                guidance_scale=args.scale,
                audio_end_in_s=SECONDS,
                generator=gen
            )

        audio = result.audios[0].detach().float().cpu().numpy()

        if audio.ndim == 2:
            audio = audio.T

        sf.write(path, audio, pipe.vae.sampling_rate)

    print('Output:', output_dir)

if __name__ == '__main__':
    main()
