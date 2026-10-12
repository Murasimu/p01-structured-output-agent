"""JSONL audit trail for every validation decision the agent makes.

Every parse error, schema violation, tool-output rejection, retry, and
success is recorded as a structured JSON line -- the evidence trail that
makes the agent debuggable and its reliability measurable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

LEVELS = ("info", "warning", "error")


class AuditLog:
    """In-memory audit log that optionally mirrors entries to a JSONL file."""

    def __init__(self, path: str | None = None) -> None:
        self.entries: list[dict[str, Any]] = []
        self.path = path

    def record(self, event: str, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Record one audit entry and append it to the JSONL file if configured."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        entry = {
            "ts": round(time.time(), 3),
            "event": event,
            "level": level,
            **fields,
        }
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    # Convenience views -------------------------------------------------
    def errors(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["level"] == "error"]

    def count(self, event: str) -> int:
        return sum(1 for e in self.entries if e["event"] == event)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e["event"]] = counts.get(e["event"], 0) + 1
        return counts

    def dump(self) -> str:
        return "\n".join(json.dumps(e, default=str) for e in self.entries)


def asdict_safe(obj: Any) -> Any:
    """Best-effort conversion for audit fields (kept tiny on purpose)."""
    try:
        return asdict(obj)
    except TypeError:
        return repr(obj)
