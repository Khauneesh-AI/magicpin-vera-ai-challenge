from __future__ import annotations

import re
from hashlib import sha1
from typing import Any


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
HOSTILE_OR_OPTOUT = (
    "stop",
    "unsubscribe",
    "not interested",
    "don't message",
    "do not message",
    "shut up",
    "abuse",
)
OFF_TOPIC = ("gst", "tax", "invoice", "salary", "rent agreement", "itr")
COMMIT = ("ok", "yes", "go ahead", "send it", "kar do", "chalega", "haan")


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def text_hash(text: str) -> str:
    return sha1(normalize(text).encode("utf-8")).hexdigest()


def is_hostile_or_opt_out(text: str) -> bool:
    lowered = normalize(text)
    return any(token in lowered for token in HOSTILE_OR_OPTOUT)


def is_off_topic(text: str) -> bool:
    lowered = normalize(text)
    return any(token in lowered for token in OFF_TOPIC)


def is_auto_reply(text: str) -> bool:
    lowered = normalize(text)
    if any(token in lowered for token in AUTO_REPLY_MARKERS):
        return True
    return bool(re.search(r"[\u0900-\u097f].*(धन्यवाद|संपर्क|जानकारी)", text))


def is_commit(text: str) -> bool:
    lowered = normalize(text)
    return any(token in lowered for token in COMMIT)


def repeated_in_history(message: str, history: list[dict[str, Any]], threshold: int = 3) -> bool:
    digest = text_hash(message)
    matching = 1
    for turn in reversed(history):
        if turn.get("from") == "bot":
            continue
        if text_hash(str(turn.get("body", ""))) == digest:
            matching += 1
        else:
            break
    return matching >= threshold


def decide_reply(
    *,
    conversation_id: str,
    message: str,
    history: list[dict[str, Any]],
    merchant_auto_reply_count: int = 0,
) -> dict[str, Any]:
    if is_hostile_or_opt_out(message):
        return {"action": "end", "rationale": "Merchant opted out or sent a hostile reply."}
    if is_off_topic(message):
        return {
            "action": "send",
            "body": "I can help with Vera growth actions here. Want me to keep this to the campaign/reply draft?",
            "cta": "binary_yes_no",
            "rationale": "Off-topic reply redirected once to Vera's merchant-growth scope.",
        }
    if is_auto_reply(message) or repeated_in_history(message, history) or merchant_auto_reply_count:
        auto_count = sum(
            1
            for turn in history
            if turn.get("from") != "bot" and is_auto_reply(str(turn.get("body", "")))
        )
        if merchant_auto_reply_count >= 3 or auto_count >= 2 or repeated_in_history(message, history):
            return {"action": "end", "rationale": "Auto-reply loop detected; conversation ended gracefully."}
        return {
            "action": "wait",
            "rationale": "Auto-reply detected; waiting avoids sending into a canned-response loop.",
        }
    if is_commit(message):
        return {
            "action": "send",
            "body": "Done. I will keep it short, specific, and ready for you to approve before sending.",
            "cta": "none",
            "rationale": "Explicit commit detected; bot switches to action confirmation.",
        }
    return {
        "action": "wait",
        "rationale": f"Low-signal reply in {conversation_id}; waiting avoids spam.",
    }
