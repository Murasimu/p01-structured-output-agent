import json

import pytest

from structured_agent import AuditLog, StructuredOutputAgent, Tool, ToolBox, ToolOutputError, extract_json
from structured_agent.agent import AgentResult
from pydantic import BaseModel, ValidationError


class Order(BaseModel):
    order_id: str
    customer: str
    total_usd: float
    items: list[str]


SCHEMA_JSON = json.dumps(Order.model_json_schema(), indent=2)


def system_msg() -> dict:
    return {"role": "system", "content": f"schema:\n{SCHEMA_JSON}"}


# ---------------------------------------------------------------- parser
class TestExtractJson:
    def test_clean_object(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced(self):
        text = "Here you go:\n```json\n{\"a\": 1}\n```\nHope that helps!"
        assert extract_json(text) == '{"a": 1}'

    def test_prose_wrapped(self):
        text = 'Sure! {"order_id": "A-1", "total": 3.5} is the object you asked for.'
        assert extract_json(text) == '{"order_id": "A-1", "total": 3.5}'

    def test_nested_with_braces_in_strings(self):
        text = 'blah {"a": {"b": "close } brace"}, "c": [1,2]} trailing'
        assert extract_json(text) == '{"a": {"b": "close } brace"}, "c": [1,2]}'

    def test_array(self):
        assert extract_json("x [1, 2, {\"k\": \"v\"}] y") == '[1, 2, {"k": "v"}]'

    def test_unbalanced_returns_none(self):
        assert extract_json('{"a": 1') is None

    def test_empty_returns_none(self):
        assert extract_json("") is None
        assert extract_json("no json here at all") is None

    def test_json_in_string_escaped(self):
        text = 'prefix {"k": "say \\" hi", "n": 2} suffix'
        assert json.loads(extract_json(text))["n"] == 2


# ---------------------------------------------------------------- agent
class ScriptedLLM:
    """Returns queued replies, records every prompt it receives."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


class TestAgent:
    def test_first_try_success(self):
        llm = ScriptedLLM(['{"order_id":"O-1","customer":"Ann","total_usd":10.0,"items":["mug"]}'])
        agent = StructuredOutputAgent(Order, llm, max_retries=3)
        result = agent.run("Extract the order")
        assert result.ok and result.attempts == 1
        assert result.value == Order(order_id="O-1", customer="Ann", total_usd=10.0, items=["mug"])

    def test_retry_after_prose_then_success(self):
        llm = ScriptedLLM([
            "Sure! The order is PO-2 for Bob, total 5 dollars, item: pen.",  # no JSON at all
            '{"order_id":"O-2","customer":"Bob","total_usd":5.0,"items":["pen"]}',
        ])
        agent = StructuredOutputAgent(Order, llm, max_retries=3)
        result = agent.run("Extract")
        assert result.ok and result.attempts == 2
        # Retry feedback must reach the model.
        assert any("failed validation" in m["content"] for c in llm.calls for m in c)

    def test_schema_violation_then_success(self):
        llm = ScriptedLLM([
            '{"order_id":"O-3","customer":"Cara","total_usd":"cheap","items":[]}',  # wrong type
            '{"order_id":"O-3","customer":"Cara","total_usd":1.5,"items":["socks"]}',
        ])
        agent = StructuredOutputAgent(Order, llm, max_retries=3)
        result = agent.run("Extract")
        assert result.ok and result.attempts == 2
        assert agent.audit.count("validation_error") == 1

    def test_max_retries_honest_failure(self):
        llm = ScriptedLLM(["no json"] * 3)
        agent = StructuredOutputAgent(Order, llm, max_retries=3)
        result = agent.run("Extract")
        assert not result.ok
        assert result.status == "max_retries_exceeded"
        assert result.value is None
        assert agent.audit.count("parse_error") == 3

    def test_audit_trail_complete(self):
        llm = ScriptedLLM(["garbage", '{"order_id":"O-9","customer":"Zed","total_usd":2.0,"items":["ink"]}'])
        audit = AuditLog()
        agent = StructuredOutputAgent(Order, llm, max_retries=3, audit=audit)
        agent.run("Extract")
        assert [e["event"] for e in audit.entries] == ["parse_error", "success"]
        assert audit.entries[0]["level"] == "error"

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError):
            StructuredOutputAgent(Order, lambda m: "", max_retries=0)


# ---------------------------------------------------------------- tools
class Args(BaseModel):
    n: int


class Out(BaseModel):
    doubled: int


class TestToolBox:
    def _box(self, audit=None) -> ToolBox:
        box = ToolBox(audit=audit)
        box.register(Tool(
            name="double",
            description="Doubles a number",
            fn=lambda a: {"doubled": a["n"] * 2},
            args_schema=Args,
            returns_schema=Out,
        ))
        return box

    def test_valid_run(self):
        out = self._box().run("double", {"n": 21})
        assert out.doubled == 42

    def test_bad_args_rejected(self):
        with pytest.raises(ValidationError):
            self._box().run("double", {"n": "not-a-number"})

    def test_bad_output_rejected_and_audited(self):
        audit = AuditLog()
        box = ToolBox(audit=audit)
        box.register(Tool(
            name="broken",
            description="Always returns garbage",
            fn=lambda a: {"oops": True},
            args_schema=Args,
            returns_schema=Out,
        ))
        with pytest.raises(ToolOutputError):
            box.run("broken", {"n": 1})
        assert audit.count("tool_output_invalid") == 1

    def test_unknown_tool(self):
        with pytest.raises(KeyError):
            self._box().run("nope", {"n": 1})

    def test_duplicate_registration(self):
        box = self._box()
        with pytest.raises(ValueError):
            box.register(Tool("double", "x", lambda a: {}, Args, Out))


# ---------------------------------------------------------------- audit
class TestAuditLog:
    def test_jsonl_file_written(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        log = AuditLog(path=str(p))
        log.record("success", attempt=1)
        log.record("parse_error", level="error", attempt=1, raw="bad")
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["event"] == "parse_error"

    def test_bad_level_rejected(self):
        with pytest.raises(ValueError):
            AuditLog().record("x", level="catastrophic")
