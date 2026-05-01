from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = 1
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickBody(BaseModel):
    now: str | None = None
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: Literal["merchant", "customer", "system"] = "merchant"
    message: str
    received_at: str | None = None
    turn_number: int = 1


class ComposedMessage(BaseModel):
    body: str
    cta: str
    send_as: Literal["vera", "merchant_on_behalf"] = "vera"
    suppression_key: str
    rationale: str


class Action(ComposedMessage):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    trigger_id: str


class ReplyDecision(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str | None = None
    cta: str | None = None
    rationale: str
