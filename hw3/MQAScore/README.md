# MQAScore

**MQAScore** measures how well a piece of music matches a text description by
asking an audio-language model (LALM) one yes/no question per musical tag and
reading off the probability of "Yes".

Global embedding similarity (CLAP, MuQ-MuLan) compresses a whole description
into one vector, which makes it insensitive to which attribute is wrong. MQAScore
instead decomposes the text into tags (genre, instrument, mood, vocals, usage
context, ...) and probes each one separately against the audio.

## How it works

1. **Tags.** The text is split into tags per attribute, either supplied by you or
   extracted with an LLM (`scripts/extract_tags.py`).
2. **One question per tag.** Each tag is placed into an attribute-specific
   template, e.g. `Can you hear the instrument 'piano' being played in this music? Answer Yes or No.`
3. **P(Yes) from logits.** For each question the LALM produces next-token logits
   at the answer position. With `logp = log_softmax(logits)` over the full vocabulary,

   ```
   P(Yes) = sigmoid( logsumexp(logp[YES]) - logsumexp(logp[NO]) )
   ```

   where `YES` / `NO` are the first token ids of `Yes, " Yes", yes, " yes", YES`
   and the corresponding "No" forms. This is the Yes mass divided by the
   Yes + No mass, a continuous score in [0, 1].
4. **Item score.** MQAScore = the unweighted mean of P(Yes) over all tags of the
   item. Per-attribute means and every per-tag score are also saved.

A **caption protocol** (`--protocol caption`) asks a single question about the whole
text instead (`Does this music match the following description? '<text>' Answer Yes or No.`),
for comparison with per-tag scoring.

## Supported models

| `--model` | Checkpoint | Env (`envs/`) | GPU memory (bf16) | Default audio length |
|---|---|---|---|---|
| `qwen3omni` | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) (thinker only) | `qwen3_omni` | ~60–70 GB (sharded across visible GPUs) | 10 s |
| `af` | [`nvidia/audio-flamingo-next-hf`](https://huggingface.co/nvidia/audio-flamingo-next-hf) | `music_flamingo` | fits one 24 GB GPU | 30 s |
| `mf` | [`nvidia/music-flamingo-hf`](https://huggingface.co/nvidia/music-flamingo-hf) | `music_flamingo` | fits one 24 GB GPU | 30 s |

**Audio length.** By default, audio is loaded mono with librosa, then zero-padded
or truncated to the model's default length before it reaches the processor.
These are the settings behind our reported numbers. Use `--fix-seconds N` to
change the length, or `--fix-seconds full` to pass the original file unchanged.

## Installation

```bash
conda env create -f envs/qwen3_omni.yml        # Qwen3-Omni + tag extraction
conda env create -f envs/music_flamingo.yml    # Audio Flamingo Next / Music Flamingo
```

See [`envs/README.md`](envs/README.md) for why two environments are needed and
for the pinned versions. Run the scripts from the repository root, or install
the package into an env with `pip install -e .`.

## Quickstart

```bash
# 0. Pick a free GPU (on multi-GPU machines use PCI_BUS_ID so ids match nvidia-smi)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

# 1. (Optional) text -> tags with an LLM, if you only have captions
conda activate qwen3_omni
python scripts/extract_tags.py --input examples/captions.jsonl --output out/tags.jsonl

# 2. Score with an LALM (per-tag MQAScore)
python scripts/score.py --model qwen3omni --input out/tags.jsonl --output out/qwen3omni.jsonl

conda activate music_flamingo
python scripts/score.py --model mf --input out/tags.jsonl --output out/mf.jsonl

# Caption protocol, for comparison
python scripts/score.py --model af --protocol caption \
    --input examples/items_caption.jsonl --output out/af_caption.jsonl
```

The example files reference `examples/audio/demo_{1,2,3}.wav`. No audio is
shipped, so add your own files there or edit the `audio` paths. Use
`--limit 10` to dry-run on a few items first.

### Python API

```python
from mqascore import MQAScorer, load_backend

scorer = MQAScorer(load_backend("qwen3omni"))   # or "af", "mf"
res = scorer.score_tags("song.wav", {"genre": ["jazz"], "instrument": ["piano", "upright bass"]})
print(res["mean_score"], res["attribute_mean"])
```

See [`examples/python_api.py`](examples/python_api.py).

## Input and output formats

**Per-tag input** (`--protocol pertag`, default). One JSON object per line:

```json
{"id": "demo_1", "audio": "audio/demo_1.wav",
 "tags": {"genre": ["jazz"], "instrument": ["piano", {"entity": "upright bass", "attributes": ["plucked"]}]}}
```

- A tag is a string or a dict with `"entity"`. The entity is what gets asked;
  other fields are kept as `extra`.
- Tags are de-duplicated case-insensitively within an attribute.
- Only attributes listed in `--attributes` are scored. The default is
  `genre,instrument,mood_theme,vocal,context`; `all` adds `rhythm_tempo`,
  `production_sound_quality` and `others`.

**Caption input** (`--protocol caption`): `{"id", "audio", "text"}`.

**Output.** One JSON object per line. Input fields are kept and these are added:

| field | meaning |
|---|---|
| `mean_score` | MQAScore (mean P(Yes) over tags) |
| `n_tags` | number of scored tags |
| `attribute_mean` | `{attribute: mean P(Yes)}` |
| `per_attribute` | `{attribute: [{tag, question, score, loglik_yes, loglik_no, extra}]}` |
| `caption_score`, `question`, `loglik_yes`, `loglik_no` | caption protocol only |

**Settings file.** Next to every output there is a `<output>.meta.json`. It records
the model id and class, Yes/No token ids, attention implementation, audio length,
system prompt and question templates. Re-running with the same `--output` resumes
where it stopped. If any recorded setting differs, the run aborts instead, so
scores from different settings never end up in one file.

## Prompts

All prompts are plain text files in [`instructions/`](instructions/), named
`instruction_<usage>.txt`:

| file | used for |
|---|---|
| `instruction_lalm_system.txt` | system prompt for every yes/no question |
| `instruction_lalm_pertag_<attribute>.txt` | question template per attribute (`{tag}` placeholder) |
| `instruction_lalm_caption.txt` | caption-protocol question (`{caption}` placeholder) |
| `instruction_llm_tag_extraction.txt` | instruction for the text LLM in `scripts/extract_tags.py` |

The code only reads these files; nothing is hard-coded. **Editing them changes
the scores.** The texts in this repo are the ones behind our results, and
`tests/test_prompts.py` fails if they change. To try other prompts without
editing the repo copy, set `MQASCORE_INSTRUCTION_DIR` to a directory with your
own versions.

## Notes

- **Batch size is always 1.** Qwen3-Omni is a mixture-of-experts model. In bf16,
  packing several questions into one batch changes expert grouping and reduction
  order. We measured max |ΔP(Yes)| = 4.1e-2 between batch=1 and packed batches,
  versus exactly 0 between two batch=1 runs. All backends therefore score one
  question per forward pass.
- **Prompt order differs by model family, as each model expects.** Qwen puts the
  audio before the question; AF/MF put the question before the audio.
- **Tag extraction** used `Qwen/Qwen3-4B-Instruct-2507` with greedy decoding. Any
  instruction-tuned LLM can be swapped in with `--model`.

## Repository layout

```
instructions/   all LALM / LLM prompts (instruction_<usage>.txt)
mqascore/       library: prompts, P(Yes) scoring, audio prep, backends, scorer, I/O
  models/       qwen3_omni.py, audio_flamingo.py (af / mf), base.py
scripts/        score.py (CLI), extract_tags.py (text -> tags)
examples/       example inputs and a Python API example (bring your own audio)
envs/           conda environment files and pinned requirements
tests/          prompt guard test; compare_reference.py for checking reproduced scores
```

## Status

This repository was extracted from our experiment code. **It has not yet been
checked numerically against the scores reported in the paper.**
`tests/compare_reference.py` is provided for that check, and the reference files
for it will go in `tests/fixtures/`.

## Citation

```bibtex
TBD
```

## License

See [LICENSE](LICENSE). Model checkpoints are subject to their own licenses on
Hugging Face.
