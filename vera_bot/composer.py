from __future__ import annotations

from typing import Any

from vera_bot.trigger_handlers import (
    customer_message,
    merchant_message,
)
from vera_bot.validators import finalize_message


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, str]:
    if customer is None and trigger.get("scope") == "customer":
        raw = customer_message(category, merchant, trigger, {})
    elif customer is not None or trigger.get("scope") == "customer":
        raw = customer_message(category, merchant, trigger, customer or {})
    else:
        raw = merchant_message(category, merchant, trigger)
    fallback_key = str(
        trigger.get("suppression_key")
        or f"{trigger.get('kind', 'unknown')}:{trigger.get('id', '')}"
    )
    return finalize_message(raw, fallback_key=fallback_key)
