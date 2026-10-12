"""Structured-output agent that makes LLMs reliable, not random.

Enforces Pydantic schemas on every LLM and tool response, retries on
parse/validation errors with targeted feedback, and keeps a JSONL audit
trail of every validation decision.
"""

from .agent import AgentResult, StructuredOutputAgent
from .audit import AuditLog
from .parser import extract_json
from .tools import Tool, ToolBox, ToolOutputError

__all__ = [
    "AgentResult",
    "StructuredOutputAgent",
    "AuditLog",
    "extract_json",
    "Tool",
    "ToolBox",
    "ToolOutputError",
]
__version__ = "1.0.0"
