"""Tag normalisation.

Accepts tags as {attribute: [tag, ...]} where each element is either a string
or a dict with an "entity" key (the format produced by scripts/extract_tags.py;
all other dict fields are kept as `extra`). Within an attribute, tags are
de-duplicated case-insensitively, keeping the first occurrence.
"""


def normalize_tags(tags, attributes):
    """Return [(attribute, tag, extra), ...] in `attributes` order."""
    out = []
    for attr in attributes:
        seen = set()
        for el in tags.get(attr) or []:
            if isinstance(el, dict):
                tag = el.get('entity')
                extra = {k: v for k, v in el.items() if k != 'entity'}
            else:
                tag, extra = el, {}
            if not isinstance(tag, str) or not tag.strip():
                continue
            tag = tag.strip()
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())
            out.append((attr, tag, extra))
    return out
