"""The core structured-output agent.

Loop: ask the LLM -> extract JSON -> validate against a Pydantic schema.
On failure, feed the *specific* error back to the model and retry, up to
`max_retries`. Every failure is logged to the audit trail. If all retries
are exhausted the agent degrades gracefully (returns a failed result,
never a hallucinated object).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .audit import AuditLog
from .parser import extract_json

T = TypeVar("T", bound=BaseModel)

LLM = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = """You are a structured-output agent.
Respond with a single JSON object that conforms EXACTLY to this JSON Schema:

{schema}

Rules:
- Output ONLY the JSON object. No markdown fences, no prose, no extra keys.
- Include every required field with the correct type.
- Where information is ambiguous, choose the most defensible value."""


@dataclass
class AgentResult:
    """Outcome of one agent run, including the full validation audit."""

    value: Any | None
    status: str  # "ok" | "max_retries_exceeded"
    attempts: int
    audit_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class StructuredOutputAgent:
    """Wraps an LLM call so callers always get schema-valid output -- or an
    honest failure with an audit trail explaining exactly what went wrong."""

    def __init__(
        self,
        schema: Type[BaseModel],
        llm: LLM,
        *,
        max_retries: int = 3,
        audit: AuditLog | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.schema = schema
        self.llm = llm
        self.max_retries = max_retries
        self.audit = audit or AuditLog()

    def run(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Ask the LLM and return a validated instance of `self.schema`.

        `context` is pre-validated data (e.g. from ToolBox) injected as a
        user message; the schema still governs the final answer.
        """
        system = SYSTEM_PROMPT.format(
            schema=json.dumps(self.schema.model_json_schema(), indent=2)
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if context is not None:
            messages.append(
                {
                    "role": "user",
                    "content": "Validated context data (already schema-checked):\n"
                    + json.dumps(context, default=str),
                }
            )
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, self.max_retries + 1):
            raw = self.llm(messages)
            candidate = extract_json(raw)

            if candidate is None:
                self.audit.record(
                    "parse_error", level="error", attempt=attempt, raw=raw[:400]
                )
                messages.extend(_retry_messages(raw, "not valid JSON"))
                continue

            try:
                value = self.schema.model_validate(json.loads(candidate))
            except (json.JSONDecodeError, ValidationError) as exc:
                self.audit.record(
                    "validation_error",
                    level="error",
                    attempt=attempt,
                    error=str(exc)[:600],
                    raw=candidate[:400],
                )
                messages.extend(_retry_messages(raw, str(exc)))
                continue

            self.audit.record("success", attempt=attempt)
            return AgentResult(
                value=value,
                status="ok",
                attempts=attempt,
                audit_entries=self.audit.entries,
            )

        self.audit.record("max_retries_exceeded", level="error", attempts=self.max_retries)
        return AgentResult(
            value=None,
            status="max_retries_exceeded",
            attempts=self.max_retries,
            audit_entries=self.audit.entries,
        )


def _retry_messages(raw: str, problem: str) -> list[dict[str, str]]:
    """Append the failed answer + targeted feedback so the model can self-correct."""
    return [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"That reply failed validation because it was {problem}. "
                "Respond again with ONLY a corrected JSON object matching the schema."
            ),
        },
    ]
