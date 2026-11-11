"""Common interface for LALM backends."""


class LALMBackend:
    """Subclasses load a model and implement `_forward(convs)`.

    Scoring is always done one question at a time (batch size 1). Batching
    changes scores: Qwen3-Omni is an MoE model and under bf16 the batch
    composition changes expert grouping and reduction order; we measured
    max |dP(Yes)| = 4.1e-2 between batch=1 and packed batches, versus 0 between
    two batch=1 runs.
    """

    name = None
    model_id = None
    default_fix_seconds = None

    def build_conv(self, question, audio_path):
        raise NotImplementedError

    def _forward(self, convs):
        """Return (p_yes, loglik_yes, loglik_no) lists for a list of conversations."""
        raise NotImplementedError

    def score(self, audio_path, questions):
        """Score each question against one audio file. Returns [(p_yes, ly, ln), ...]."""
        out = []
        for q in questions:
            p, ly, ln = self._forward([self.build_conv(q, audio_path)])
            out.append((float(p[0]), float(ly[0]), float(ln[0])))
        return out

    def protocol(self):
        """Model-side settings recorded in the output's .meta.json."""
        return {'model': self.name, 'model_id': self.model_id,
                'model_class': type(self.model).__name__,
                'yes_ids': self.yes_ids, 'no_ids': self.no_ids,
                'batch_size': 1}
