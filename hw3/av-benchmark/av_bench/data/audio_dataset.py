import logging
from pathlib import Path
from typing import List

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def load_audio(path):
    audio, sample_rate = sf.read(
        str(path),
        dtype='float32',
        always_2d=True,
    )
    waveform = torch.from_numpy(audio.T.copy())
    return waveform, sample_rate

# from https://github.com/haoheliu/audioldm_eval
class AudioDataset(Dataset):

    def __init__(
        self,
        datalist: List[Path],
        audio_length: float = 8.0,
        sr: int = 16000,
        limit_num=None,
    ):
        self.datalist = datalist
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self.sr = sr
        self.audio_length = audio_length

        self.resampler = {}

    def __getitem__(self, idx: int):
        while True:
            try:
                filename = self.datalist[idx]
                waveform = self.read_from_file(filename)
                break
            except Exception as e:
                log.error(f'Error loading {self.datalist[idx]}: {e}')
                idx = (idx + 1) % len(self.datalist)

        return waveform, filename.stem

    def __len__(self):
        return len(self.datalist)

    def read_from_file(self, audio_file):
        waveform, sample_rate = load_audio(audio_file)
        waveform = waveform.mean(dim=0)  # mono
        waveform = waveform - waveform.mean()

        if sample_rate == self.sr:
            audio = waveform
        else:
            if sample_rate not in self.resampler:
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sr,
                )
            audio = self.resampler[sample_rate](waveform)

        audio = audio[:int(self.sr * self.audio_length)].unsqueeze(0)

        return audio
