from __future__ import annotations

import re
import time
from typing import Any

from fastapi import APIRouter

from vera_bot import __version__, config
from vera_bot.composer import compose
from vera_bot.reply_handler import decide_reply
from vera_bot.schemas import ContextBody, ReplyBody, TickBody
from vera_bot.store import Store

router = APIRouter()
store = Store()
STARTED_AT = time.time()


def conversation_id(merchant_id: str, trigger_id: str, customer_id: str | None) -> str:
    base = f"conv_{customer_id or merchant_id}_{trigger_id}"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", base)[:120]


def action_from_message(
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    message: dict[str, str],
    customer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trigger_id = str(trigger.get("id") or trigger.get("trigger_id") or "")
    merchant_id = str(merchant.get("merchant_id") or trigger.get("merchant_id") or "")
    customer_id = None
    if customer:
        customer_id = str(customer.get("customer_id") or trigger.get("customer_id") or "")
    elif trigger.get("customer_id"):
        customer_id = str(trigger.get("customer_id"))
    template_base = "merchant" if message["send_as"] == "merchant_on_behalf" else "vera"
    return {
        "conversation_id": conversation_id(merchant_id, trigger_id, customer_id),
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "trigger_id": trigger_id,
        "template_name": f"{template_base}_{trigger.get('kind', 'message')}_v1",
        "template_params": [message["body"][:120]],
        **message,
    }


@router.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 3),
        "contexts_loaded": store.count_contexts(),
    }


@router.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "team_name": config.TEAM_NAME,
        "team_members": config.TEAM_MEMBERS,
        "contact_email": config.CONTACT_EMAIL,
        "model": config.MODEL_NAME,
        "approach": config.APPROACH,
        "version": __version__,
    }


@router.post("/context")
def push_context(body: ContextBody) -> dict[str, Any]:
    return store.push_context(body.scope, body.context_id, body.version, body.payload)


@router.post("/tick")
def tick(body: TickBody) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for trigger_id in body.available_triggers:
        if len(actions) >= config.MAX_ACTIONS_PER_TICK:
            break
        trigger = store.get_payload("trigger", trigger_id)
        if not trigger:
            continue
        merchant_id = trigger.get("merchant_id")
        merchant = store.get_payload("merchant", merchant_id)
        if not merchant:
            continue
        category_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category")
        category = store.get_payload("category", category_slug)
        if not category:
            continue
        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = store.get_payload("customer", customer_id)
        suppression_key = str(trigger.get("suppression_key") or trigger_id)
        if store.was_sent(str(merchant_id), suppression_key):
            continue
        message = compose(category, merchant, trigger, customer)
        if store.seen_recent_body(str(merchant_id), message["body"]):
            continue
        action = action_from_message(merchant, trigger, message, customer)
        actions.append(action)
        store.mark_sent(str(merchant_id), message["suppression_key"])
        store.remember_body(str(merchant_id), message["body"])
        store.add_turn(
            action["conversation_id"],
            {"from": "bot", "body": message["body"], "trigger_id": action["trigger_id"]},
        )
    return {"actions": actions}


@router.post("/reply")
def reply(body: ReplyBody) -> dict[str, Any]:
    history = store.get_history(body.conversation_id)
    store.add_turn(
        body.conversation_id,
        {"from": body.from_role, "body": body.message, "turn_number": body.turn_number},
    )
    decision = decide_reply(
        conversation_id=body.conversation_id,
        message=body.message,
        history=history[:-1],
    )
    if decision["action"] == "send":
        store.add_turn(body.conversation_id, {"from": "bot", "body": decision.get("body", "")})
    return decision
