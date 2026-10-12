"""Demo: extract a structured Order from messy, real-world-ish email text.

First run the deterministic pipeline (LLM responses are mocked), showing
parse-error retry and schema-violation retry, then run the same agent
against a live LLM if LLM_API_KEY is configured.

Usage:
  python demo.py                 # scripted demo only (no network)
  python demo.py --live          # + one live LLM extraction
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from structured_agent import AuditLog, StructuredOutputAgent
from structured_agent.llm import make_llm


class Order(BaseModel):
    order_id: str
    customer: str
    total_usd: float
    items: list[str]
    currency: str = "USD"
    notes: str = ""


EMAIL = """Subject: Re: our purchase

Hey team -- ref PO-8841 here. Ann Lee from Meridian Foods.
We'd like: 2x enamel mug, 1x pine desk, 3x floor lamp.
Invoice total is 642.50 dollars. Please confirm by Friday!
Ship to the usual warehouse. Thanks!
"""


def scripted_llm(replies: list[str]):
    def llm(messages):
        return replies.pop(0)
    return llm


def banner(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


def main() -> int:
    # -- Scenario 1: model wraps JSON in prose, then corrects itself -------
    banner("Scenario 1: LLM wraps JSON in chatter -> retry succeeds")
    audit = AuditLog()
    agent = StructuredOutputAgent(
        Order,
        scripted_llm([
            "Sure! Here's your order info:\n```json\n"
            '{"order_id": "PO-8841", "customer": "Ann Lee", "total_usd": 642.50,'
            ' "items": ["enamel mug", "pine desk", "floor lamp"], "notes": "ship usual warehouse"}\n```'
            "\nHope that helps!",
            '{"order_id": "PO-8841", "customer": "Ann Lee", "total_usd": 642.50,'
            ' "items": ["enamel mug", "pine desk", "floor lamp"], "currency": "USD",'
            ' "notes": "ship usual warehouse"}',
        ]),
        max_retries=3,
        audit=audit,
    )
    r = agent.run("Extract the purchase order from the email below.\n\n" + EMAIL)
    print(f"status={r.status}  attempts={r.attempts}")
    if r.ok:
        print(f"order  = {r.value.order_id} for {r.value.customer}")
        print(f"total  = ${r.value.total_usd:.2f}  items={r.value.items}")
    for e in audit.entries:
        print(f"[audit] {e['level']:7} {e['event']}")

    # -- Scenario 2: wrong type -> schema-violation retry ------------------
    banner("Scenario 2: schema violation (total as string) -> targeted retry")
    audit2 = AuditLog()
    agent2 = StructuredOutputAgent(
        Order,
        scripted_llm([
            '{"order_id": "PO-8841", "customer": "Ann Lee", "total_usd": "642 dollars", "items": []}',
            '{"order_id": "PO-8841", "customer": "Ann Lee", "total_usd": 642.5,'
            ' "items": ["enamel mug", "pine desk", "floor lamp"]}',
        ]),
        max_retries=3,
        audit=audit2,
    )
    r2 = agent2.run("Extract the order.\n\n" + EMAIL)
    print(f"status={r2.status}  attempts={r2.attempts}")
    print(f"[audit] {[e['event'] for e in audit2.entries]}")

    # -- Scenario 3: live LLM (optional) ----------------------------------
    if "--live" in sys.argv:
        banner("Scenario 3: live LLM extraction")
        try:
            live = make_llm()
        except RuntimeError as exc:
            print(f"skipping live demo: {exc}")
            return 0
        audit3 = AuditLog()
        agent3 = StructuredOutputAgent(Order, live, max_retries=4, audit=audit3)
        r3 = agent3.run("Extract the purchase order from this email:\n\n" + EMAIL)
        print(f"status={r3.status}  attempts={r3.attempts}")
        if r3.ok:
            print(r3.value.model_dump_json(indent=2))
        for e in audit3.entries:
            print(f"[audit] {e['level']:7} {e['event']}")

    return 0 if r.ok and r2.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
