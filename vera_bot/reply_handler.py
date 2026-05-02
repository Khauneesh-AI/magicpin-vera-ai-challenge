"""Two-tier reply handler: keyword fast-path + LLM classification.

Tier 1 (deterministic, ~0ms): catches repeated auto-replies and exact
opt-out keywords without any LLM call.

Tier 2 (LLM, ~1-2s): classifies ambiguous replies via GPT-5.4-mini
and routes to the appropriate action.
"""

from __future__ import annotations

import logging
import re
from hashlib import sha1
from typing import Any

from vera_bot import llm_client
from vera_bot.prompts import (
    _CLASSIFY_SYSTEM,
    build_classify_user,
    build_reply_compose_system,
    build_reply_compose_user,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword constants for Tier 1 fast-path
# ---------------------------------------------------------------------------

AUTO_REPLY_MARKERS = (
    "thank you for contacting",
    "thanks for contacting",
    "we will get back",
    "away from whatsapp",
    "out of office",
    "auto reply",
    "automatic reply",
    "aapki jaankari",
    "bahut-bahut shukriya",
    "hum jald",
    "dhanyavaad",
)

EXACT_OPT_OUT = ("stop", "unsubscribe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _text_hash(text: str) -> str:
    return sha1(_normalize(text).encode("utf-8")).hexdigest()


def is_auto_reply_keyword(text: str) -> bool:
    lowered = _normalize(text)
    if any(token in lowered for token in AUTO_REPLY_MARKERS):
        return True
    return bool(re.search(r"[\u0900-\u097f].*(धन्यवाद|संपर्क|जानकारी)", text))


def is_exact_opt_out(text: str) -> bool:
    lowered = _normalize(text)
    return lowered in EXACT_OPT_OUT


def repeated_in_history(
    message: str, history: list[dict[str, Any]], threshold: int = 3
) -> bool:
    digest = _text_hash(message)
    matching = 1
    for turn in reversed(history):
        if turn.get("from") == "bot":
            continue
        if _text_hash(str(turn.get("body", ""))) == digest:
            matching += 1
        else:
            break
    return matching >= threshold


# ---------------------------------------------------------------------------
# Keyword-only fallback (used when LLM is unavailable)
# ---------------------------------------------------------------------------

_HOSTILE_KEYWORDS = (
    "stop", "unsubscribe", "not interested", "don't message",
    "do not message", "shut up", "band karo",
)
_OFF_TOPIC_KEYWORDS = ("gst", "tax", "invoice", "salary", "rent agreement", "itr")
_COMMIT_KEYWORDS = ("ok", "yes", "go ahead", "send it", "kar do", "chalega", "haan")


def _keyword_fallback(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    lowered = _normalize(message)
    if any(k in lowered for k in _HOSTILE_KEYWORDS):
        return {"action": "end", "rationale": "Opt-out detected (keyword fallback)."}
    if any(k in lowered for k in _OFF_TOPIC_KEYWORDS):
        return {
            "action": "send",
            "body": "I can help with your Google profile and customer outreach here. Want to get back to that?",
            "cta": "binary_yes_no",
            "rationale": "Off-topic redirect (keyword fallback).",
        }
    if is_auto_reply_keyword(message):
        return {"action": "wait", "rationale": "Auto-reply detected (keyword fallback)."}
    if any(k in lowered for k in _COMMIT_KEYWORDS):
        return {
            "action": "send",
            "body": "Done. I will keep it short, specific, and ready for you to approve before sending.",
            "cta": "none",
            "rationale": "Commit detected (keyword fallback).",
        }
    return {"action": "wait", "rationale": "Low-signal reply (keyword fallback)."}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def handle_reply(
    conversation_id: str,
    message: str,
    history: list[dict[str, Any]],
    merchant_id: str | None = None,
    merchant_context: dict[str, Any] | None = None,
    trigger_context: dict[str, Any] | None = None,
    auto_reply_count: int = 0,
    detected_language: str = "en",
) -> dict[str, Any]:
    """Classify a merchant reply and return the bot's next action.

    Returns a dict with at minimum {"action": "send"|"wait"|"end"}.
    When action is "send", also includes "body", "cta", "rationale".
    """

    # ---- Tier 1: Deterministic fast-path ----
    if auto_reply_count >= 3 or repeated_in_history(message, history):
        return {"action": "end", "rationale": "Auto-reply loop detected; exiting."}

    if is_exact_opt_out(message):
        return {"action": "end", "rationale": "Explicit opt-out."}

    # ---- Tier 2: LLM classification ----
    try:
        classify_user = build_classify_user(message, history)
        classification = await llm_client.classify_reply(
            _CLASSIFY_SYSTEM, classify_user
        )
    except Exception:
        logger.warning("LLM classify failed, using keyword fallback", exc_info=True)
        return _keyword_fallback(message, history)

    intent = classification.intent
    action = classification.action

    # Auto-reply routing
    if intent == "auto_reply":
        if auto_reply_count == 0:
            return {"action": "wait", "rationale": "First auto-reply; waiting for human."}
        return {"action": "end", "rationale": "Repeated auto-reply; exiting gracefully."}

    # Opt-out
    if intent == "opt_out":
        return {"action": "end", "rationale": "Opt-out detected by LLM."}

    # Off-topic — redirect once
    if intent == "off_topic":
        return {
            "action": "send",
            "body": "I can help with your Google profile and customer outreach here. Want to get back to that?",
            "cta": "binary_yes_no",
            "rationale": "Off-topic redirect.",
        }

    # Commit / question / engaged — compose a reply
    if action == "send":
        try:
            reply_system = build_reply_compose_system(merchant_context, history, detected_language)
            reply_user = build_reply_compose_user(message, intent)
            reply_msg = await llm_client.compose_reply(reply_system, reply_user)
            return {
                "action": "send",
                "body": reply_msg.body,
                "cta": reply_msg.cta,
                "rationale": f"{intent.replace('_', ' ').capitalize()} detected; {reply_msg.rationale}",
            }
        except Exception:
            logger.warning("LLM reply compose failed, using keyword fallback", exc_info=True)
            return _keyword_fallback(message, history)

    # Default: wait
    return {"action": "wait", "rationale": f"LLM classified as {intent}; waiting."}
