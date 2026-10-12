"""Robust JSON extraction from raw LLM output.

LLMs wrap JSON in markdown fences, prepend apologies, or append prose.
Instead of trusting them, we scan for the first *balanced* JSON value.
"""

from __future__ import annotations

import json


def _balanced(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced `opener..closer` span in `text`, if any."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> str | None:
    """Extract the first syntactically valid JSON object or array from `text`.

    Returns the JSON substring, or None if nothing parseable is found.
    Objects are preferred over arrays; candidates must survive json.loads.
    """
    if not text:
        return None
    # Consider both object and array openers, keep the earliest balanced
    # span that survives json.loads (objects at the same index win).
    best: tuple[int, str] | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        candidate = _balanced(text, opener, closer)
        if candidate is None:
            continue
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        pos = text.find(candidate)
        if best is None or pos < best[0]:
            best = (pos, candidate)
    return best[1] if best else None
