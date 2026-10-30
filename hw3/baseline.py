import argparse
import torch
import soundfile as sf

from pathlib import Path
from tqdm import tqdm
from diffusers import StableAudioPipeline

from common import read_captions

MODEL_PATH = "stabilityai/stable-audio-open-1.0"
CAPTIONS_PATH = Path("test.csv")
OUTPUT_DIR = Path("outputs/generated")

SEED = 42
SECONDS = 20

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--scale', type=float, default=7.0)
    return parser.parse_args()

def main():
    args = parse_args()

    dtype = torch.float16
    device = "cuda"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_captions(CAPTIONS_PATH)
    pipe = StableAudioPipeline.from_pretrained(MODEL_PATH, dtype=dtype)
    pipe = pipe.to(device)

    for sample_id, caption in tqdm(rows, desc='Generate'):
        path = OUTPUT_DIR / f'{sample_id}.wav'

        seed = SEED + int(sample_id) if sample_id.isdecimal() else SEED
        gen = torch.Generator(device='cpu').manual_seed(seed)

        result = pipe(
            prompt=caption,
            negative_prompt='',
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            audio_end_in_s=SECONDS,
            generator=gen
        )

        audio = result.audios[0].detach().float().cpu().numpy()

        if audio.ndim == 2:
            audio = audio.T

        sf.write(path, audio, pipe.vae.sampling_rate)

    print('Output:', OUTPUT_DIR.resolve())

if __name__ == '__main__':
    main()
