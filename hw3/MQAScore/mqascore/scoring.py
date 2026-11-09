"""Yes/No probability from next-token logits.

P(Yes) = sigmoid( logsumexp(logp[YES_IDS]) - logsumexp(logp[NO_IDS]) )

where logp is log_softmax over the full vocabulary at the position that would
generate the answer, and YES_IDS / NO_IDS are the first token ids of common
surface forms of "Yes" / "No". The normalising constant cancels, so P(Yes) is
the Yes mass divided by (Yes mass + No mass). log-likelihoods are returned as
well so other aggregations can be computed later.
"""
import torch

YES_WORDS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_WORDS = ["No", " No", "no", " no", "NO"]


def first_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def yes_no_ids(tokenizer):
    return first_token_ids(tokenizer, YES_WORDS), first_token_ids(tokenizer, NO_WORDS)


def pyes_from_logits(next_logits, yes_t, no_t):
    """next_logits: [B, V]. Returns lists (p_yes, loglik_yes, loglik_no)."""
    logp = torch.log_softmax(next_logits.float(), dim=-1)
    ly = torch.logsumexp(logp[:, yes_t], dim=-1)
    ln = torch.logsumexp(logp[:, no_t], dim=-1)
    pyes = torch.sigmoid(ly - ln)
    return pyes.cpu().tolist(), ly.cpu().tolist(), ln.cpu().tolist()
