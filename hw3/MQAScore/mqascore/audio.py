"""Audio preparation.

Our reported numbers use fixed-length inputs: the clip is loaded with librosa
(default 22050 Hz, mono), zero-padded or truncated to `fix_seconds`, written to
a cache wav, and that file is handed to the model's processor (which resamples
to its own rate). Defaults per backend: Qwen3-Omni 10 s, AF-Next / Music
Flamingo 30 s. `fix_seconds=None` passes the original file unchanged
(full length).
"""
import hashlib
from pathlib import Path


def prepare_audio(src, fix_seconds, cache_dir):
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    if fix_seconds is None:
        return str(src)
    import librosa
    import numpy as np
    import soundfile as sf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:12]
    out = cache_dir / f"{src.stem}_{key}_fix{fix_seconds:g}.wav"
    if not out.exists():
        wav, sr = librosa.load(str(src), mono=True)
        target = int(round(fix_seconds * sr))
        if len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        else:
            wav = wav[:target]
        tmp = out.with_name(out.stem + ".tmp.wav")
        sf.write(str(tmp), wav, sr, format="WAV")
        tmp.replace(out)
    return str(out)
