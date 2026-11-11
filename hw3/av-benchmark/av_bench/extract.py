import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from av_bench.data.audio_dataset import AudioDataset
from av_bench.extraction_models import ExtractionModels

log = logging.getLogger(__name__)


@torch.inference_mode()
def extract(audio_path: Path,
            output_path: Path,
            *,
            audio_length: float = 8.0,
            batch_size: int = 128,
            num_workers: int = 32,
            device: str,
            skip_video_related: bool = False,
            skip_clap: bool = False):

    audios = sorted(list(audio_path.glob('*.wav')) + list(audio_path.glob('*.flac')),
                    key=lambda x: x.stem)

    log.info(f'{len(audios)} audios found in {audio_path}')

    models = ExtractionModels().to(device).eval()
    dataset = AudioDataset(audios, audio_length=audio_length, sr=16000)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # extract features with VGGish
    out_dict = {}
    for wav, filename in tqdm(loader):
        wav = wav.squeeze(1).float()
        features = models.vggish(wav).cpu()
        for i, f_name in enumerate(filename):
            out_dict[f_name] = features[i]

    output_path.mkdir(parents=True, exist_ok=True)
    vggish_feature_path = output_path / 'vggish_features.pth'
    log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
    torch.save(out_dict, vggish_feature_path)
    del out_dict

    # PassT, need 32K sampling rate
    dataset = AudioDataset(audios, audio_length=audio_length, sr=32_000)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out_logits = {}
    for wav, filename in tqdm(loader):
        wav = wav.squeeze(1).float().to(device)

        if (wav.size(-1) >= 320000):
            wav = wav[..., :320000]
        else:
            wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

        features = models.passt_model(wav).cpu()
        # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
        # logits is 527 dim; features is 768
        logits = features[:, :527]
        for i, f_name in enumerate(filename):
            out_logits[f_name] = logits[i]
    output_path.mkdir(parents=True, exist_ok=True)
    passt_feature_path = output_path / 'passt_logits.pth'
    log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
    torch.save(out_logits, passt_feature_path)
