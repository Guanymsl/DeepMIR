"""Backend registry. Imports are lazy so each backend only needs its own env."""

MODELS = {
    "af2": "nvidia/audio-flamingo-2-0.5B",
}

def load_backend(name, **kwargs):
    if name == "af2":
        from .af2 import AudioFlamingo2Backend
        return AudioFlamingo2Backend(**kwargs)
    raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
