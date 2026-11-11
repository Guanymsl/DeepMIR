#!/usr/bin/env python
"""Extract structured tags from text with an instruction-tuned LLM (vLLM).

This is the optional first step when you have captions/prompts but no tags.
The instruction is instructions/instruction_llm_tag_extraction.txt; we used
Qwen/Qwen3-4B-Instruct-2507 with greedy decoding.

Input JSONL : {"id": "...", "text": "...", ...}   (other fields, e.g. "audio", are kept)
Output JSONL: input fields + {"tags": {genre, instrument, mood_theme, rhythm_tempo,
              vocal, production_sound_quality, context, others}}
Items whose LLM output cannot be parsed go to <output>.failures.jsonl; rerun
to retry them (items already in --output are skipped).

Example:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python scripts/extract_tags.py --input examples/captions.jsonl --output out/tags.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mqascore.io import append_jsonl, done_ids, read_jsonl  # noqa: E402
from mqascore.prompts import ALL_ATTRIBUTES, tag_extraction_instruction  # noqa: E402

CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def extract_json(text):
    """Parse the LLM output into a tag dict, or raise ValueError.

    Strips markdown code fences and falls back to the outermost {...} span.
    Missing categories become []; unexpected keys are dropped.
    """
    cleaned = CODE_FENCE_RE.sub('', text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find('{'), cleaned.rfind('}')
        if start == -1 or end <= start:
            raise ValueError('no JSON object found in output')
        obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f'parsed JSON is {type(obj).__name__}, not an object')
    if not any(k in obj for k in ALL_ATTRIBUTES):
        raise ValueError('none of the expected keys present')
    return {k: obj.get(k, []) for k in ALL_ATTRIBUTES}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--temperature', type=float, default=0.0, help='0.0 = greedy')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fail_path = out.with_name(out.name + '.failures.jsonl')

    items = read_jsonl(args.input)
    done = done_ids(out)
    todo = [it for it in items if it['id'] not in done][: args.limit]
    print(f'{len(items)} items, {len(done)} already done, {len(todo)} to process')
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    instruction = tag_extraction_instruction()
    convs = [[{'role': 'system', 'content': instruction},
              {'role': 'user', 'content': it['text']}] for it in todo]
    outputs = llm.chat(
        convs,
        params,
        chat_template_kwargs={'enable_thinking': False},
    )

    if fail_path.exists():
        fail_path.unlink()
    n_fail = 0
    for it, o in zip(todo, outputs):
        raw = o.outputs[0].text
        try:
            append_jsonl(out, {**it, 'tags': extract_json(raw)})
        except (ValueError, json.JSONDecodeError) as e:
            n_fail += 1
            append_jsonl(fail_path, {**it, 'error': str(e), 'raw_output': raw})
    print(f'{len(todo) - n_fail} ok, {n_fail} failed' + (f' (see {fail_path})' if n_fail else ''))
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
