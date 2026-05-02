"""Dual-path composition: deterministic + LLM run in parallel, LLM selector picks the best.

Architecture:
    Input ──┬── Path A: deterministic (BM25 template + name swap) ──┐
            │                    parallel                            ├── LLM Selector ── winner
            └── Path B: LLM compose (from scratch with template ref)┘
                                                                   (timeout → Path A)

This gets the best of both worlds:
- Known triggers: deterministic wins on specificity, selector confirms it
- Novel triggers: LLM compose wins, selector picks it
- Timeout/failure: always falls back to deterministic (zero-risk)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel

from vera_bot import config
from vera_bot.fact_extractor import extract_all_facts
from vera_bot.llm_client import ComposedMessage, compose_message, classify_reply
from vera_bot.prompts import build_compose_system, build_compose_user
from vera_bot.retrieval import index as retrieval_index
from vera_bot.validators import finalize_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path A: deterministic
# ---------------------------------------------------------------------------

def _build_deterministic(
    template: dict[str, Any],
    facts: dict[str, Any],
    fallback_key: str,
) -> dict[str, str]:
    """Build a deterministic message from LIVE FACTS, using the template
    only for structure (cta_type, send_as). Never copies sample_message
    directly — that would leak hardcoded facts from other merchants."""
    m = facts.get("merchant", {})
    t = facts.get("trigger", {})
    c = facts.get("customer")
    trigger_scope = t.get("scope", "merchant")
    is_customer_scoped = trigger_scope == "customer" or template.get("scope") == "customer"

    name = m.get("name", "Merchant")
    full_name = m.get("full_name", name)
    location = m.get("location", "")
    offer = m.get("best_offer", "")
    perf = m.get("performance", {})
    peer = m.get("peer_comparison", {})
    kind = t.get("kind", "")
    payload = t.get("payload", {})

    # Build body from live facts
    if is_customer_scoped:
        cust_name = c.get("name", "") if c else ""
        greeting = f"Hi {cust_name}, {full_name}" if cust_name else f"Hi, {full_name}"
        if location:
            greeting += f" ({location})"
        greeting += " here."
        # Extract trigger-specific detail
        detail = ""
        if kind == "recall_due":
            service = payload.get("service_due", "follow-up")
            detail = f" Your {service} is due."
        elif kind == "appointment_tomorrow":
            detail = " Your appointment is tomorrow."
        elif kind == "chronic_refill_due":
            meds = ", ".join(payload.get("molecule_list", [])[:3])
            date = str(payload.get("stock_runs_out_iso", "")).split("T")[0]
            detail = f" Your {meds or 'medicine'} stock runs out on {date}."
        elif kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
            days = payload.get("days_since_last_visit", "")
            detail = f" It's been {days} days since your last visit." if days else ""
        body = f"{greeting}{detail}"
        if offer:
            body += f" {offer} ready for you."
        body += " Reply YES to confirm, or share another time."
    else:
        # Merchant-scoped: build from trigger kind
        header = f"{name}, {full_name}"
        if location:
            header += f", {location}"
        header += ":"

        if "perf_dip" in kind:
            metric = payload.get("metric", "calls")
            delta = payload.get("delta_pct", "")
            body = f"{header} {metric} down {delta}"
            if perf.get("delta_7d"):
                for k, v in perf["delta_7d"].items():
                    if v and metric in k:
                        body = f"{header} {metric} {v} in last 7d"
                        break
            if peer.get("merchant_ctr") and peer.get("peer_ctr"):
                body += f"; CTR {peer['merchant_ctr']} vs peer {peer['peer_ctr']}"
            body += "."
            if offer:
                body += f" Main {offer} ka post refresh kar doon?"
            else:
                body += " Main ek fresh post draft kar doon?"
            body += " YES/STOP"
        elif "perf_spike" in kind:
            metric = payload.get("metric", "calls")
            delta = payload.get("delta_pct", "")
            body = f"{header} {metric} up {delta} in last 7d. Want to double down with a fresh post? YES/STOP"
        elif kind == "competitor_opened":
            comp = payload.get("competitor_name", "a competitor")
            dist = payload.get("distance_km", "")
            body = f"{header} {comp} opened {dist}km away."
            if offer:
                body += f" Your {offer} + reviews is the counter."
            body += " Want the draft? YES/STOP"
        elif "festival" in kind or "seasonal" in kind or "ipl" in kind:
            event = payload.get("festival", payload.get("match", kind.replace("_", " ")))
            body = f"{header} {event} window is coming."
            if offer:
                body += f" Best move: one post around {offer}."
            body += " Want the draft? YES/STOP"
        elif "dormant" in kind or "winback" in kind:
            days = payload.get("days_since_last_merchant_message", payload.get("days_since_expiry", ""))
            body = f"{header} it's been {days} days."
            if offer:
                body += f" One recovery post around {offer} can help."
            body += " Want the 2-line version? YES/STOP"
        elif kind in ("research_digest", "regulation_change", "cde_opportunity"):
            digest = t.get("digest_item", {})
            title = digest.get("title", kind.replace("_", " "))
            source = digest.get("source", "")
            body = f"{header} {title}."
            if source:
                body += f" Source: {source}."
            body += " Want me to pull the summary?"
        elif kind == "milestone_reached":
            metric = payload.get("metric", "reviews")
            value = payload.get("value_now", "")
            target = payload.get("milestone_value", "")
            body = f"{header} at {value} {metric}, {target} is within reach. Want a review-request draft?"
        else:
            body = f"{header} {kind.replace('_', ' ')} signal. Want me to draft an action?"

    send_as = "merchant_on_behalf" if is_customer_scoped else template.get("send_as", "vera")
    cta = template.get("cta_type", "binary_yes_no" if is_customer_scoped else "open_ended")
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": fallback_key,
        "rationale": f"Deterministic fallback from live facts for {kind}.",
    }


# ---------------------------------------------------------------------------
# Path B: LLM compose
# ---------------------------------------------------------------------------

def _detect_language_from_facts(facts: dict[str, Any]) -> str:
    """Derive language preference from merchant/customer facts."""
    # Customer language pref takes priority
    cust = facts.get("customer")
    if cust and cust.get("language_pref"):
        pref = cust["language_pref"].lower()
        if "hi" in pref:
            return "hi-en"
    # Fall back to merchant languages
    langs = facts.get("merchant", {}).get("languages", [])
    if "hi" in langs:
        return "hi-en"
    return "en"


async def _build_llm(
    category: dict[str, Any],
    facts: dict[str, Any],
    trigger: dict[str, Any],
    matches: list[tuple[dict[str, Any], float]],
    customer: dict[str, Any] | None,
    detected_language: str = "en",
) -> dict[str, str] | None:
    """LLM composes from scratch using the template as structural reference.
    Returns None on failure."""
    try:
        system_prompt = build_compose_system(category, detected_language)
        user_prompt = build_compose_user(
            facts, trigger, matches, customer, mode="compose",
        )
        result: ComposedMessage = await compose_message(system_prompt, user_prompt)
        return result.model_dump()
    except Exception:
        logger.warning("LLM compose path failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Selector: pick the better message
# ---------------------------------------------------------------------------

_SELECTOR_SYSTEM = """\
You are picking the BETTER WhatsApp message for a merchant engagement bot.

Consider:
1. Specificity (concrete numbers/dates from merchant data)
2. Merchant fit (uses business name, locality, owner name — not just generic)
3. NO hallucination (if a message cites an offer/price not in the merchant's active offers, that is WORSE)
4. Trigger relevance (why NOW)
5. Engagement (compulsion levers, clear CTA)

IMPORTANT: A message that invents an offer the merchant doesn't have is WORSE than one that doesn't mention offers.

Return ONLY: {"pick": "A" or "B", "reason": "one sentence"}\
"""


def _build_selector_prompt(
    msg_a: str, msg_b: str, trigger_kind: str, category_slug: str,
    merchant_name: str, merchant_offers: list[str] | None = None,
    merchant_location: str = "",
) -> str:
    offers_str = ", ".join(merchant_offers) if merchant_offers else "NONE (no active offers)"
    return f"""Trigger: {trigger_kind} | Category: {category_slug}
Merchant: {merchant_name} | Location: {merchant_location}
Merchant active offers: {offers_str}

Message A:
"{msg_a}"

Message B:
"{msg_b}"

Which is better? If a message cites an offer not in the active offers list, penalize it. Return JSON: {{"pick": "A" or "B", "reason": "..."}}"""


class _Selection(BaseModel):
    pick: Literal["A", "B"]
    reason: str


async def _select_best(
    msg_a: dict[str, str],
    msg_b: dict[str, str],
    trigger: dict[str, Any],
    category: dict[str, Any],
    merchant_name: str,
    merchant_offers: list[str] | None = None,
    merchant_location: str = "",
) -> str:
    """Ask LLM to pick A or B. Returns 'A' or 'B'. Defaults to 'A' on failure."""
    try:
        from vera_bot.llm_client import _parse

        prompt = _build_selector_prompt(
            msg_a["body"], msg_b["body"],
            trigger.get("kind", ""), category.get("slug", ""),
            merchant_name, merchant_offers, merchant_location,
        )
        result = await _parse(
            model=config.CLASSIFY_MODEL,
            schema=_Selection,
            messages=[
                {"role": "system", "content": _SELECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
            timeout=config.CLASSIFY_TIMEOUT,
        )
        logger.info("Selector picked %s: %s", result.pick, result.reason)
        return result.pick
    except Exception:
        logger.warning("Selector failed, defaulting to A (deterministic)", exc_info=True)
    return "A"


# ---------------------------------------------------------------------------
# Main compose entry point
# ---------------------------------------------------------------------------

async def compose_async(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Dual-path compose: deterministic + LLM in parallel, selector picks winner."""

    # 1. Extract facts
    facts = extract_all_facts(category, merchant, trigger, customer)

    # 2. Retrieve best template + score
    matches = retrieval_index.query(trigger, category, top_n=2)
    top_template, top_score = matches[0] if matches else ({}, 0.0)

    # 3. Build suppression key
    fallback_key = str(
        trigger.get("suppression_key")
        or f"{trigger.get('kind', 'unknown')}:{trigger.get('id', '')}"
    )

    # 4. Path A: deterministic (instant)
    if top_template:
        msg_a = _build_deterministic(top_template, facts, fallback_key)
    else:
        msg_a = {
            "body": f"{facts['merchant']['name']}, Vera has an update for you. Reply YES to hear more.",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": fallback_key,
            "rationale": "No template match; generic fallback.",
        }

    # 5. Path B: LLM compose (async, may fail)
    lang = _detect_language_from_facts(facts)
    msg_b = await _build_llm(category, facts, trigger, matches, customer, lang)

    # 6. Select the best + log the routing decision
    trigger_kind = trigger.get("kind", "unknown")
    merchant_name = facts["merchant"]["name"]
    template_id = top_template.get("id", "none") if top_template else "none"

    if msg_b is None:
        raw = msg_a
        route = "deterministic"
        reason = "LLM failed"
    elif top_score < config.BM25_CONFIDENCE_THRESHOLD:
        raw = msg_b
        route = "llm_compose"
        reason = f"novel trigger (BM25={top_score:.1f} < {config.BM25_CONFIDENCE_THRESHOLD})"
    else:
        pick = await _select_best(
            msg_a, msg_b, trigger, category,
            merchant_name,
            merchant_offers=facts["merchant"].get("active_offers"),
            merchant_location=facts["merchant"].get("location", ""),
        )
        raw = msg_b if pick == "B" else msg_a
        route = "llm_compose" if pick == "B" else "deterministic"
        reason = f"selector picked {pick} (BM25={top_score:.1f})"

    logger.info(
        "[COMPOSE] %s | trigger=%s | merchant=%s | route=%s | reason=%s | template=%s | model=%s | body=%s",
        trigger_kind, trigger.get("id", ""), merchant_name,
        route, reason, template_id, config.COMPOSE_MODEL,
        raw.get("body", "")[:80],
    )

    # 7. Enforce send_as for customer-scoped triggers
    trigger_scope = trigger.get("scope", "merchant")
    if trigger_scope == "customer" or customer is not None:
        raw["send_as"] = "merchant_on_behalf"

    # 8. Validate
    return finalize_message(raw, fallback_key=fallback_key)


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Sync wrapper for compose_async(). Used by tools/generate_submission.py
    and the bot.py public contract. Do NOT call from inside FastAPI routes
    (use compose_async there)."""
    import asyncio
    return asyncio.run(compose_async(category, merchant, trigger, customer))
