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

## Supported models

| `--model` | Checkpoint | Env (`envs/`) | GPU memory (bf16) | Default audio length |
|---|---|---|---|---|
| `af2` | [`nvidia/audio-flamingo-2-0.5B`](https://huggingface.co/nvidia/audio-flamingo-2-0.5B) | `mqascore_af2` | depends on device | 10 s |

**Audio length.** By default, audio is loaded mono with librosa, then zero-padded
or truncated to the model's default length before it reaches the processor.
These are the settings behind our reported numbers. Use `--fix-seconds N` to
change the length, or `--fix-seconds full` to pass the original file unchanged.

## Installation

Run the scripts from the repository root using the HW3 environments:
`mqascore_tags` for text tag extraction and `mqascore_af2` for AF2 scoring.
The scripts use local imports; no editable package installation is required.
AF2 sources are loaded from `../AudioFlamingo2/inference_HF_pretrained/`.

## Quickstart

From `hw3/`, use the wrapper:

```bash
conda run --no-capture-output -n deepmir_hw3 python evaluate.py --pred baseline --metrics mqascore
```

Or run these commands from the MQAScore repository root with your input JSONL:

```bash
conda run --no-capture-output -n mqascore_tags python scripts/extract_tags.py --model Qwen/Qwen3-1.7B --input captions.jsonl --output tags.jsonl
conda run --no-capture-output -n mqascore_af2 python scripts/score.py --model af2 --input tags.jsonl --output scores.jsonl --protocol pertag --fix-seconds 10
```

Use `--limit 10` to dry-run on a few items first.

### Python API

```python
from mqascore import MQAScorer, load_backend

scorer = MQAScorer(load_backend("af2"))
res = scorer.score_tags("song.wav", {"genre": ["jazz"], "instrument": ["piano", "upright bass"]})
print(res["mean_score"], res["attribute_mean"])
```


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


**Output.** One JSON object per line. Input fields are kept and these are added:

| field | meaning |
|---|---|
| `mean_score` | MQAScore (mean P(Yes) over tags) |
| `n_tags` | number of scored tags |
| `attribute_mean` | `{attribute: mean P(Yes)}` |
| `per_attribute` | `{attribute: [{tag, question, score, loglik_yes, loglik_no, extra}]}` |

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
| `instruction_llm_tag_extraction.txt` | instruction for the text LLM in `scripts/extract_tags.py` |

The code only reads these files; nothing is hard-coded. **Editing them changes
the scores.** To try other prompts without
editing the repo copy, set `MQASCORE_INSTRUCTION_DIR` to a directory with your
own versions.

## Notes

- **Batch size is always 1.** All questions are scored one per forward pass.
- **Tag extraction** uses `Qwen/Qwen3-1.7B` in the HW3 wrapper. Any instruction-tuned LLM can be swapped in with `--model`.
- Keep extracted tags fixed when comparing audio models.

## Repository layout

```
instructions/   all LALM / LLM prompts (instruction_<usage>.txt)
mqascore/       library: prompts, P(Yes) scoring, audio prep, backends, scorer, I/O
  models/       af2.py, base.py
scripts/        score.py (CLI), extract_tags.py (text -> tags)
```

## License

See [LICENSE](LICENSE). Model checkpoints are subject to their own licenses on
Hugging Face.
