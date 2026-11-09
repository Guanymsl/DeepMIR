"""MQAScore: text-to-music alignment by per-tag yes/no probing of audio-language models."""
from .models import MODELS, load_backend
from .prompts import ALL_ATTRIBUTES, DEFAULT_ATTRIBUTES
from .scorer import MQAScorer

__all__ = ["MQAScorer", "load_backend", "MODELS", "ALL_ATTRIBUTES", "DEFAULT_ATTRIBUTES"]
__version__ = "0.1.0"
