#!/usr/bin/env python
"""Score audio/tag JSONL items with the AF2 per-tag MQAScore protocol.

Input fields: id, audio, tags. Existing output IDs are resumed.
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
    p.add_argument("--protocol", choices=["pertag"], default="pertag")
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES),
                   help=f"comma list, or 'all' for {ALL_ATTRIBUTES}")
    p.add_argument("--fix-seconds", default="default",
                   help="pad/truncate audio to N seconds; 'full' = no change; "
                        "'default' = AF2 default (10 seconds)")
    p.add_argument("--cache-dir", default=".mqascore_audio_cache")
    p.add_argument("--model-id", default=None, help="override the HF model id / local path")
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
    key = "tags"
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
    backend = load_backend(args.model, **kw)
    fs = {"default": {}, "full": {"fix_seconds": None}}.get(
        args.fix_seconds, {"fix_seconds": float(args.fix_seconds)})
    scorer = MQAScorer(backend, attributes=attrs, cache_dir=args.cache_dir, **fs)
    check_protocol(out, scorer.protocol(args.protocol))

    from tqdm import tqdm
    failed = []
    for it in tqdm(todo):
        try:
            res = scorer.score_tags(it["audio"], it["tags"])
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
