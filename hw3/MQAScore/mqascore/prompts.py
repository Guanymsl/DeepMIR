"""Prompt loading. Every prompt sent to an LALM or LLM lives in `instructions/`
as `instruction_<usage>.txt`; this module only reads those files and fills
placeholders. Nothing is hard-coded here, so the text files are the single
source of truth.

Files:
  instruction_lalm_system.txt              system prompt for yes/no probing
  instruction_lalm_pertag_<attribute>.txt  one question template per attribute ({tag})
  instruction_lalm_caption.txt             whole-caption question ({caption})
  instruction_llm_tag_extraction.txt       caption -> tags instruction for the text LLM

LALM files are stored with exactly one trailing newline, which is stripped on
load. The LLM instruction is passed through verbatim.
"""
import os
from functools import lru_cache
from pathlib import Path

INSTRUCTION_DIR = Path(os.environ.get(
    "MQASCORE_INSTRUCTION_DIR",
    Path(__file__).resolve().parent.parent / "instructions"))

# All attributes that have a per-tag question template.
ALL_ATTRIBUTES = [
    "genre", "instrument", "mood_theme", "rhythm_tempo", "vocal",
    "production_sound_quality", "context", "others",
]
# Default attribute set for item-level MQAScore (the five axes used in our
# MusicEval experiments). Pass `attributes=ALL_ATTRIBUTES` to use all eight.
DEFAULT_ATTRIBUTES = ["genre", "instrument", "mood_theme", "vocal", "context"]


@lru_cache(maxsize=None)
def load_instruction(usage, strip_newline=True):
    path = INSTRUCTION_DIR / f"instruction_{usage}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing instruction file: {path}")
    text = path.read_text(encoding="utf-8")
    if strip_newline and text.endswith("\n"):
        text = text[:-1]
    return text


def system_prompt():
    return load_instruction("lalm_system")


def question_template(attribute):
    if attribute not in ALL_ATTRIBUTES:
        raise KeyError(f"unknown attribute {attribute!r}; choose from {ALL_ATTRIBUTES}")
    return load_instruction(f"lalm_pertag_{attribute}")


def pertag_question(attribute, tag):
    return question_template(attribute).format(tag=tag)


def caption_question(caption):
    """The caption is stripped, never truncated."""
    return load_instruction("lalm_caption").format(caption=caption.strip())


def tag_extraction_instruction():
    return load_instruction("llm_tag_extraction", strip_newline=False)


def text_protocol(granularity, attributes=None):
    """Text-side settings recorded in the output's .meta.json."""
    p = {"system_prompt": system_prompt(), "granularity": granularity}
    if granularity == "pertag":
        p["attributes"] = list(attributes)
        p["question_templates"] = {a: question_template(a) for a in attributes}
    else:
        p["caption_question_template"] = load_instruction("lalm_caption")
    return p


if __name__ == "__main__":
    print(f"instruction dir: {INSTRUCTION_DIR}")
    print(f"system : {system_prompt()}")
    for a in ALL_ATTRIBUTES:
        print(f"{a:<26}: {pertag_question(a, 'jazz')}")
    print(f"caption: {caption_question('A lively summer piano solo piece.')}")
    print(f"tag-extraction instruction: {len(tag_extraction_instruction())} chars")
