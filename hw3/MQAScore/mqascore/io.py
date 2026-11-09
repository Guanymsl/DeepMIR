"""JSONL I/O, resume, and the .meta.json settings guard.

Every output file gets a sibling `<output>.meta.json` recording the full
setting (model, prompts, audio length, ...). Resuming into an existing output
with a different setting aborts, so scores from different settings never mix
in one file.
"""
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def done_ids(path):
    path = Path(path)
    return {r["id"] for r in read_jsonl(path)} if path.exists() else set()


def meta_path(out_path):
    out_path = Path(out_path)
    return out_path.with_name(out_path.name + ".meta.json")


def check_protocol(out_path, protocol):
    out_path, mp = Path(out_path), meta_path(out_path)
    if out_path.exists() and mp.exists():
        prev = json.loads(mp.read_text())
        diff = sorted(k for k in set(prev) | set(protocol) if prev.get(k) != protocol.get(k))
        if diff:
            lines = "\n".join(f"  {k}: existing={prev.get(k)!r} now={protocol.get(k)!r}"
                              for k in diff)
            raise SystemExit(f"{out_path} was produced with different settings:\n{lines}\n"
                             f"Use a new --output to score with the new settings.")
    elif out_path.exists():
        print(f"[warn] {out_path} exists without {mp.name}; cannot verify its settings.")
    mp.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
