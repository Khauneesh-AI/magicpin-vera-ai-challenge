from __future__ import annotations

import re

from vera_bot import config

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
REQUIRED_KEYS = ("body", "cta", "send_as", "suppression_key", "rationale")


def _clean_text(text: object, *, max_chars: int = config.MAX_BODY_CHARS) -> str:
    cleaned = " ".join(str(text or "").replace("None", "").split())
    cleaned = URL_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" ,;:-")
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "."


def finalize_message(message: dict, *, fallback_key: str) -> dict[str, str]:
    body = _clean_text(message.get("body"))
    cta = _clean_text(message.get("cta"), max_chars=80) or "open_ended"
    send_as = str(message.get("send_as") or "vera")
    if send_as not in config.ALLOWED_SEND_AS:
        send_as = "vera"
    suppression_key = _clean_text(message.get("suppression_key"), max_chars=160) or fallback_key
    rationale = _clean_text(message.get("rationale"), max_chars=240)
    if not rationale:
        rationale = "Deterministic Vera message composed from available context."
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


def is_contract_message(message: dict) -> bool:
    return all(key in message for key in REQUIRED_KEYS)
