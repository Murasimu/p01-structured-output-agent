"""Typed tool registry: tool outputs are schema-validated before an agent
ever consumes them.

A tool that returns garbage is just as dangerous as a hallucinating LLM.
Every tool declares an args schema and a returns schema (Pydantic models);
`ToolBox.run` validates both sides and rejects bad output loudly instead
of letting it silently poison the agent's context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .audit import AuditLog


class ToolOutputError(ValueError):
    """Raised when a tool's output violates its declared returns schema."""


class UnknownToolError(KeyError):
    """Raised when asked to run a tool that is not registered."""


@dataclass(frozen=True)
class Tool:
    """A callable with declared, validated input and output schemas."""

    name: str
    description: str
    fn: Callable[[dict[str, Any]], Any]
    args_schema: type[BaseModel]
    returns_schema: type[BaseModel]


@dataclass
class ToolBox:
    """Registry of tools with two-sided schema validation."""

    tools: dict[str, Tool] = field(default_factory=dict)
    audit: AuditLog | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self.tools[tool.name] = tool

    def run(self, name: str, args: dict[str, Any]) -> BaseModel:
        """Validate args, execute, validate output; never return unvalidated data."""
        if name not in self.tools:
            raise UnknownToolError(name)
        tool = self.tools[name]
        validated_args = tool.args_schema.model_validate(args)

        raw = tool.fn(validated_args.model_dump())

        try:
            return tool.returns_schema.model_validate(raw)
        except ValidationError as exc:
            if self.audit is not None:
                self.audit.record(
                    "tool_output_invalid",
                    level="error",
                    tool=name,
                    error=str(exc)[:600],
                    raw=str(raw)[:400],
                )
            raise ToolOutputError(f"tool {name!r} produced invalid output: {exc}") from exc

    def describe(self) -> str:
        """Human/LLM-readable capability listing."""
        return "\n".join(
            f"- {t.name}: {t.description} (returns: {t.returns_schema.__name__})"
            for t in self.tools.values()
        )
