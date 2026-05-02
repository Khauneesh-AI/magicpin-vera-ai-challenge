"""Pure-function fact extraction from context dicts.

Refactored from the old trigger_handlers.py helpers. Every function takes
raw context dicts and returns a plain dict of extracted facts — no LLM, no
f-string composition, no side effects.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def first_name(merchant: dict[str, Any]) -> str:
    ident = merchant.get("identity", {})
    owner = ident.get("owner_first_name") or ident.get("name", "Merchant")
    owner = str(owner).strip()
    category = merchant.get("category_slug", "")
    if category == "dentists" and not owner.lower().startswith("dr"):
        return f"Dr. {owner}"
    return owner


def merchant_short_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("name") or first_name(merchant)


def city_locality(merchant: dict[str, Any]) -> str:
    ident = merchant.get("identity", {})
    locality = ident.get("locality")
    city = ident.get("city")
    if locality and city:
        return f"{locality}, {city}"
    return locality or city or ""


def merchant_languages(merchant: dict[str, Any]) -> list[str]:
    return merchant.get("identity", {}).get("languages", [])


# ---------------------------------------------------------------------------
# Offer helpers
# ---------------------------------------------------------------------------

def active_offer(merchant: dict[str, Any], category: dict[str, Any] | None = None) -> str:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return str(offer["title"])
    if category:
        for offer in category.get("offer_catalog", []):
            if offer.get("title"):
                return str(offer["title"])
    return ""


def all_active_offers(merchant: dict[str, Any], category: dict[str, Any] | None = None) -> list[str]:
    offers: list[str] = []
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            offers.append(str(offer["title"]))
    if not offers and category:
        for offer in category.get("offer_catalog", []):
            if offer.get("title"):
                offers.append(str(offer["title"]))
    return offers


# ---------------------------------------------------------------------------
# Performance / peer stats
# ---------------------------------------------------------------------------

def peer_ctr(category: dict[str, Any]) -> float | None:
    stats = category.get("peer_stats", {})
    value = stats.get("avg_ctr")
    return float(value) if isinstance(value, (int, float)) else None


def performance_summary(merchant: dict[str, Any]) -> dict[str, Any]:
    perf = merchant.get("performance", {})
    ctr_raw = perf.get("ctr")
    ctr_display = f"{ctr_raw * 100:.1f}%" if isinstance(ctr_raw, (int, float)) else ctr_raw
    delta_7d = perf.get("delta_7d", {})
    delta_display = {}
    for key, val in delta_7d.items():
        formatted = _fmt_pct(val, signed=True)
        delta_display[key] = formatted if formatted else val
    return {
        "views": perf.get("views"),
        "calls": perf.get("calls"),
        "directions": perf.get("directions"),
        "ctr": ctr_display,
        "ctr_raw": ctr_raw,
        "delta_7d": delta_display,
    }


def get_peer_comparison(merchant: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    perf = merchant.get("performance", {})
    stats = category.get("peer_stats", {})
    merchant_ctr = perf.get("ctr")
    peer = stats.get("avg_ctr")
    m_display = f"{merchant_ctr * 100:.1f}%" if isinstance(merchant_ctr, (int, float)) else merchant_ctr
    p_display = f"{peer * 100:.1f}%" if isinstance(peer, (int, float)) else peer
    return {
        "merchant_ctr": m_display,
        "peer_ctr": p_display,
        "ctr_vs_peer": (
            "below" if merchant_ctr and peer and merchant_ctr < peer
            else "above" if merchant_ctr and peer and merchant_ctr > peer
            else "unknown"
        ),
        "peer_avg_rating": stats.get("avg_rating"),
        "peer_avg_reviews": stats.get("avg_reviews"),
    }


# ---------------------------------------------------------------------------
# Digest / knowledge items
# ---------------------------------------------------------------------------

def find_digest_item(category: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    payload = trigger.get("payload", {})
    wanted = (
        payload.get("top_item_id")
        or payload.get("digest_item_id")
        or payload.get("alert_id")
        or payload.get("item_id")
    )
    digest = category.get("digest", [])
    if wanted:
        for item in digest:
            if item.get("id") == wanted:
                return item
    if digest:
        kind = trigger.get("kind", "")
        if "compliance" in kind or "regulation" in kind:
            for item in digest:
                if item.get("kind") in {"compliance", "regulation"}:
                    return item
        if "research" in kind:
            for item in digest:
                if item.get("kind") == "research":
                    return item
        return digest[0]
    return {}


# ---------------------------------------------------------------------------
# Customer helpers
# ---------------------------------------------------------------------------

def customer_name(customer: dict[str, Any] | None) -> str:
    if not customer:
        return ""
    name = str(customer.get("identity", {}).get("name") or "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def customer_state(customer: dict[str, Any] | None) -> str:
    if not customer:
        return ""
    return customer.get("state", "")


def customer_language_pref(customer: dict[str, Any] | None) -> str:
    if not customer:
        return ""
    return str(customer.get("identity", {}).get("language_pref", "")).lower()


# ---------------------------------------------------------------------------
# Review themes / signals
# ---------------------------------------------------------------------------

def top_review_theme(merchant: dict[str, Any]) -> dict[str, Any]:
    themes = merchant.get("review_themes", [])
    if not themes:
        return {}
    theme = themes[0]
    return {
        "theme": theme.get("theme", ""),
        "occurrences": theme.get("occurrences_30d"),
        "sentiment": theme.get("sentiment"),
    }


def merchant_signals(merchant: dict[str, Any]) -> list[str]:
    return merchant.get("signals", [])


# ---------------------------------------------------------------------------
# Formatting helpers (prevent LLM misreading raw floats)
# ---------------------------------------------------------------------------

def _fmt_pct(value: Any, signed: bool = False) -> str | None:
    """Convert raw decimal (0.5 -> 50%) or pass-through ints."""
    if not isinstance(value, (int, float)):
        return None
    n = value * 100 if abs(value) <= 1 else value
    sign = "+" if signed and n > 0 else ""
    return f"{sign}{n:.0f}%"


def _humanize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Pre-format trigger payload values so the LLM sees human-readable strings.

    Converts raw decimals to percentages, strips internal IDs, etc.
    """
    out: dict[str, Any] = {}
    skip_keys = {"merchant_id", "customer_id", "category", "digest_item_id",
                 "top_item_id", "alert_id", "item_id"}
    for key, value in payload.items():
        if key in skip_keys:
            continue
        if "pct" in key or "delta" in key:
            formatted = _fmt_pct(value, signed=True)
            if formatted:
                out[key] = formatted
                continue
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Aggregate extractors (for prompt building)
# ---------------------------------------------------------------------------

def extract_merchant_facts(merchant: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": first_name(merchant),
        "full_name": merchant_short_name(merchant),
        "location": city_locality(merchant),
        "languages": merchant_languages(merchant),
        "category_slug": merchant.get("category_slug", category.get("slug", "")),
        "subscription": merchant.get("subscription", {}),
        "performance": performance_summary(merchant),
        "peer_comparison": get_peer_comparison(merchant, category),
        "active_offers": all_active_offers(merchant, category),
        "best_offer": active_offer(merchant, category),
        "signals": merchant_signals(merchant),
        "review_theme": top_review_theme(merchant),
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "verified": merchant.get("identity", {}).get("verified"),
    }


def extract_trigger_facts(trigger: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    payload = trigger.get("payload", {})
    digest_kinds = {
        "research_digest", "regulation_change", "cde_opportunity",
        "category_research_digest_release",
    }
    digest = find_digest_item(category, trigger) if trigger.get("kind", "") in digest_kinds else {}
    # Resolve digest into human-readable fields (no raw IDs)
    digest_clean = {}
    if digest:
        digest_clean = {
            "title": digest.get("title", ""),
            "source": digest.get("source", ""),
            "date": digest.get("date", ""),
            "summary": digest.get("summary", ""),
            "actionable": digest.get("actionable", ""),
            "trial_n": digest.get("trial_n"),
            "patient_segment": str(digest.get("patient_segment", "")).replace("_", " "),
            "credits": digest.get("credits"),
        }
        # Remove None/empty values
        digest_clean = {k: v for k, v in digest_clean.items() if v}
    return {
        "trigger_id": trigger.get("id", ""),
        "kind": trigger.get("kind", ""),
        "scope": trigger.get("scope", "merchant"),
        "source": trigger.get("source", ""),
        "urgency": trigger.get("urgency"),
        "suppression_key": trigger.get("suppression_key", ""),
        "payload": _humanize_payload(payload),
        "digest_item": digest_clean,
    }


def extract_customer_facts(customer: dict[str, Any] | None) -> dict[str, Any] | None:
    if not customer:
        return None
    return {
        "name": customer_name(customer),
        "state": customer_state(customer),
        "language_pref": customer_language_pref(customer),
        "relationship": customer.get("relationship", {}),
        "preferences": customer.get("preferences", {}),
        "consent": customer.get("consent", {}),
    }


def extract_all_facts(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "merchant": extract_merchant_facts(merchant, category),
        "trigger": extract_trigger_facts(trigger, category),
        "customer": extract_customer_facts(customer),
        "category_slug": category.get("slug", ""),
        "category_voice": category.get("voice", {}),
    }
