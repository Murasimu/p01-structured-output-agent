# Structured Output Agent

**Project 1 / 12 — Agentic AI portfolio.**
*Shows: you can make LLMs reliable, not random.*

A minimal, dependency-light agent that guarantees schema-valid output from
any LLM (or tool), with targeted self-correction retries and a full JSONL
audit trail of every validation decision.

## Why this matters

Raw LLM output is unreliable: models wrap JSON in markdown fences, prepend
apologies, invent extra keys, or silently change types. In production that
means crashes, corrupted records, and silent data drift. This project shows
the three defenses that fix it:

1. **Enforce** — every response must validate against a Pydantic model.
2. **Retry with feedback** — the exact validation error is fed back to the
   model, so it can correct itself (usually in one extra attempt).
3. **Audit** — every parse error, schema violation, and retry is logged as
   structured JSONL, so reliability is measurable, not anecdotal.

The same discipline is applied to *tools*: every tool declares an args
schema and a returns schema, and outputs that fail validation are rejected
loudly instead of poisoning the agent's context.

## Architecture

```
caller ──run(prompt)──> StructuredOutputAgent
                          │  build system prompt from schema.model_json_schema()
                          ▼
                        LLM (any messages->text callable)
                          │
                          ▼
                        extract_json()      robust balanced-brace scanner
                          │
                     ┌────┴─────┐
                  valid        invalid ──> audit log ──> feedback message ─┐
                     │                                                      │
                     ▼                                                      │
               return AgentResult(ok)                retry loop (max N) ◄──┘
                                          exhausted -> AgentResult(failed, honest)
```

- `structured_agent/agent.py` — retry loop, graceful degradation
- `structured_agent/parser.py` — balanced-brace JSON extraction (no regex fragility)
- `structured_agent/tools.py` — two-sided schema-validated tool registry
- `structured_agent/audit.py` — JSONL audit trail (events, levels, file sink)
- `structured_agent/llm.py` — OpenAI-compatible live LLM wiring (demo-only)

## Quickstart

```bash
uv venv && uv pip install -e ".[demo]"   # or: pip install pydantic openai
python demo.py                            # deterministic demo, no network
python demo.py --live                     # + one live LLM call
```

For the live demo, set `LLM_API_KEY` (plus optional `LLM_BASE_URL` and
`LLM_MODEL`) — works with OpenAI, Groq, DeepSeek, vLLM, llama.cpp servers,
or any OpenAI-compatible gateway. Copy `.env.example` to `.env`.

## Tests

```bash
uv pip install pytest && pytest -q
```

24 tests cover: JSON extraction edge cases (fences, prose, nested braces,
unbalanced input), first-try success, prose-wrapped retry, schema-violation
retry, honest failure after max retries, audit trail completeness, and
two-sided tool validation.

## Design decisions

- **Balanced-brace scanner over regex** — regex JSON extraction breaks on
  nested quotes/braces; a tiny state machine handles `"close } brace"` in
  strings correctly.
- **Feed back the specific error** — generic "try again" wastes retries;
  the Pydantic error text tells the model exactly which field to fix.
- **Fail honestly** — when retries are exhausted the agent returns
  `status="max_retries_exceeded"` with `value=None` rather than guessing.
- **Framework-free core** — the agent depends only on Pydantic; the LLM is
  any `messages -> text` callable, so it plugs into any stack.

## Portfolio context

Part of a 12-project agentic-AI portfolio. See the profile README for the
full list: structured output, RAG with citations, ReAct planning, tool
orchestration, memory, HITL approval, cost routing, event automation,
multi-agent debate, self-eval, production observability, and OSS
contribution.
