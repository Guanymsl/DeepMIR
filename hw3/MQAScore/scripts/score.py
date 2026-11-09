#!/usr/bin/env python
"""Score (audio, text) items with MQAScore.

Input JSONL, one item per line:
  pertag : {"id": "...", "audio": "path.wav", "tags": {"genre": ["jazz"], "instrument": [...], ...}}
  caption: {"id": "...", "audio": "path.wav", "text": "A calm jazz piano piece ..."}
Tags may be plain strings or {"entity": ...} dicts (output of extract_tags.py).

Output JSONL, one line per item (input fields are kept, scores added), plus
<output>.meta.json with the full setting. Re-running with the same --output
resumes; changing any setting with the same --output aborts.

Example:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python scripts/score.py --model qwen3omni \
      --input examples/items_pertag.jsonl --output out/qwen3omni_pertag.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mqascore import ALL_ATTRIBUTES, DEFAULT_ATTRIBUTES, MODELS, MQAScorer, load_backend  # noqa: E402
from mqascore.io import append_jsonl, check_protocol, done_ids, read_jsonl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocol", choices=["pertag", "caption"], default="pertag")
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES),
                   help=f"comma list, or 'all' for {ALL_ATTRIBUTES}")
    p.add_argument("--fix-seconds", default="default",
                   help="pad/truncate audio to N seconds; 'full' = no change; "
                        "'default' = backend default (qwen3omni 10, af/mf 30)")
    p.add_argument("--cache-dir", default=".mqascore_audio_cache")
    p.add_argument("--model-id", default=None, help="override the HF model id / local path")
    p.add_argument("--attn-impl", default="auto", help="qwen3omni only: auto|flash_attention_2|sdpa")
    p.add_argument("--limit", type=int, default=None, help="score only the first N items (dry run)")
    return p.parse_args()


def main():
    args = parse_args()
    attrs = ALL_ATTRIBUTES if args.attributes == "all" else [
        a.strip() for a in args.attributes.split(",") if a.strip()]
    bad = [a for a in attrs if a not in ALL_ATTRIBUTES]
    if bad:
        raise SystemExit(f"unknown attributes {bad}; choose from {ALL_ATTRIBUTES}")

    items = read_jsonl(args.input)[: args.limit]
    key = "tags" if args.protocol == "pertag" else "text"
    missing = [it.get("id") for it in items if "id" not in it or "audio" not in it or key not in it]
    if missing:
        raise SystemExit(f"{len(missing)} items lack id/audio/{key}, e.g. {missing[:3]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(out)
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} already in {out}, {len(todo)} to score")
    if not todo:
        return

    kw = {"model_id": args.model_id} if args.model_id else {}
    if args.model == "qwen3omni":
        kw["attn_impl"] = args.attn_impl
    backend = load_backend(args.model, **kw)
    fs = {"default": {}, "full": {"fix_seconds": None}}.get(
        args.fix_seconds, {"fix_seconds": float(args.fix_seconds)})
    scorer = MQAScorer(backend, attributes=attrs, cache_dir=args.cache_dir, **fs)
    check_protocol(out, scorer.protocol(args.protocol))

    from tqdm import tqdm
    failed = []
    for it in tqdm(todo):
        try:
            if args.protocol == "pertag":
                res = scorer.score_tags(it["audio"], it["tags"])
            else:
                res = scorer.score_caption(it["audio"], it["text"])
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[skip] {it['id']}: {type(e).__name__}: {e}", flush=True)
            failed.append(it["id"])
            continue
        append_jsonl(out, {**it, "scorer": args.model, "granularity": args.protocol, **res})

    print(f"wrote {len(todo) - len(failed)} items to {out}")
    if failed:
        print(f"{len(failed)} items failed and were not written; rerun to retry: {failed[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
