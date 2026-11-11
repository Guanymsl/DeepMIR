"""MQAScore: per-tag yes/no probing of an audio-language model.

For an (audio, tags) pair, every tag is turned into an attribute-specific
yes/no question, the LALM's P(Yes) is computed from next-token logits, and the
item score is the unweighted mean over all tags. A caption protocol (one
question about the whole text) is also provided for comparison.
"""
from .audio import prepare_audio
from .prompts import DEFAULT_ATTRIBUTES, pertag_question, text_protocol
from .tags import normalize_tags

_UNSET = object()


class MQAScorer:
    def __init__(self, backend, attributes=None, fix_seconds=_UNSET,
                 cache_dir=".mqascore_audio_cache"):
        """
        backend:     a loaded backend from `mqascore.models.load_backend`.
        attributes:  attributes to score (default: DEFAULT_ATTRIBUTES).
        fix_seconds: pad/truncate audio to this length; None = full length;
                     unset = the backend's default (Qwen3-Omni 10, AF/MF 30).
        """
        self.backend = backend
        self.attributes = list(attributes or DEFAULT_ATTRIBUTES)
        self.fix_seconds = backend.default_fix_seconds if fix_seconds is _UNSET else fix_seconds
        self.cache_dir = cache_dir

    def _audio(self, audio_path):
        return prepare_audio(audio_path, self.fix_seconds, self.cache_dir)

    def score_tags(self, audio_path, tags):
        """tags: {attribute: [str | {"entity": str, ...}]}. Returns a result dict."""
        flat = normalize_tags(tags, self.attributes)
        questions = [pertag_question(a, t) for a, t, _ in flat]
        scores = self.backend.score(self._audio(audio_path), questions) if questions else []
        per_attribute = {}
        for (attr, tag, extra), q, (p, ly, ln) in zip(flat, questions, scores):
            per_attribute.setdefault(attr, []).append(
                {"tag": tag, "question": q, "score": p,
                 "loglik_yes": ly, "loglik_no": ln, "extra": extra})
        all_scores = [s[0] for s in scores]
        return {
            "n_tags": len(all_scores),
            "mean_score": sum(all_scores) / len(all_scores) if all_scores else None,
            "attribute_mean": {a: sum(x["score"] for x in v) / len(v)
                               for a, v in per_attribute.items()},
            "per_attribute": per_attribute,
        }

    def protocol(self, granularity):
        p = self.backend.protocol()
        p.update(text_protocol(granularity, self.attributes))
        p["fix_seconds"] = self.fix_seconds
        p["audio_loading"] = ("original file" if self.fix_seconds is None else
                              "librosa.load(mono, sr=22050) + zero-pad/truncate")
        return p
