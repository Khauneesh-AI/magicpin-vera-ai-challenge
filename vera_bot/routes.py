"""FastAPI routes — async with parallel LLM composition, cadence planning,
nudge tracking, and per-turn language detection."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, Response

from vera_bot import __version__, config
from vera_bot.composer import compose_async
from vera_bot.reply_handler import handle_reply, is_auto_reply_keyword
from vera_bot.schemas import ContextBody, ReplyBody, TickBody
from vera_bot.store import Store

logger = logging.getLogger(__name__)

router = APIRouter()
store = Store()
STARTED_AT = time.time()


def _conversation_id(merchant_id: str, trigger_id: str, customer_id: str | None) -> str:
    base = f"conv_{customer_id or merchant_id}_{trigger_id}"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", base)[:120]


def _action_from_message(
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
        "conversation_id": _conversation_id(merchant_id, trigger_id, customer_id),
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "trigger_id": trigger_id,
        "template_name": f"{template_base}_{trigger.get('kind', 'message')}_v1",
        "template_params": [message["body"][:120]],
        **message,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 3),
        "contexts_loaded": store.count_contexts(),
    }


@router.head("/healthz")
def healthz_head() -> Response:
    return Response(status_code=200)


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
async def tick(body: TickBody) -> dict[str, Any]:
    pending: list[tuple[dict, dict, dict, dict | None]] = []

    for trigger_id in body.available_triggers[:config.MAX_ACTIONS_PER_TICK]:
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

        # --- Cadence check: max 8 messages per 24h per merchant ---
        # (nudge check only applies to /v1/reply, not proactive ticks —
        #  each trigger is a fresh conversation, not an unanswered follow-up)
        if store.get_sends_in_window(str(merchant_id), 24) >= 8:
            logger.info("Skipping %s — cadence limit (8/24h)", merchant_id)
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = store.get_payload("customer", customer_id)
        suppression_key = str(trigger.get("suppression_key") or trigger_id)
        if store.was_sent(str(merchant_id), suppression_key):
            continue
        pending.append((category, merchant, trigger, customer))

    if not pending:
        return {"actions": []}

    # Fire all LLM compositions in parallel
    results = await asyncio.gather(
        *[compose_async(cat, mer, trg, cust) for cat, mer, trg, cust in pending],
        return_exceptions=True,
    )

    actions: list[dict[str, Any]] = []
    for (cat, mer, trg, cust), result in zip(pending, results):
        if isinstance(result, BaseException):
            logger.error("Tick compose error: %s", result)
            continue
        merchant_id = str(mer.get("merchant_id", ""))
        if store.seen_recent_body(merchant_id, result["body"]):
            continue
        action = _action_from_message(mer, trg, result, cust)
        actions.append(action)
        store.mark_sent(merchant_id, result["suppression_key"])
        store.remember_body(merchant_id, result["body"])
        store.record_bot_send(merchant_id)
        store.record_send_time(merchant_id)
        store.add_turn(
            action["conversation_id"],
            {"from": "bot", "body": result["body"], "trigger_id": action["trigger_id"]},
        )

    return {"actions": actions}


@router.post("/reply")
async def reply(body: ReplyBody) -> dict[str, Any]:
    history = store.get_history(body.conversation_id)
    store.add_turn(
        body.conversation_id,
        {"from": body.from_role, "body": body.message, "turn_number": body.turn_number},
    )

    # --- Per-turn language detection ---
    detected_lang = store.detect_and_store_language(body.conversation_id, body.message)

    # --- Reset nudge counter on merchant reply ---
    if body.merchant_id:
        store.record_merchant_reply(body.merchant_id)

    # Track auto-reply count per merchant
    auto_reply_count = 0
    if body.merchant_id and is_auto_reply_keyword(body.message):
        auto_reply_count = store.remember_auto_reply(body.merchant_id, body.message)

    # Load merchant context for reply composition
    merchant_context = None
    if body.merchant_id:
        merchant_context = store.get_payload("merchant", body.merchant_id)

    decision = await handle_reply(
        conversation_id=body.conversation_id,
        message=body.message,
        history=history[:-1],
        merchant_id=body.merchant_id,
        merchant_context=merchant_context,
        auto_reply_count=auto_reply_count,
        detected_language=detected_lang,
    )

    if decision.get("action") == "send":
        store.add_turn(
            body.conversation_id,
            {"from": "bot", "body": decision.get("body", "")},
        )
        # Track nudge for cadence
        if body.merchant_id:
            store.record_bot_send(body.merchant_id)

    return decision
