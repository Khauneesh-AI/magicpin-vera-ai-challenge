"""Prompt templates for LLM composition and classification.

7 prompt families for composition, each with a different reasoning strategy:
  knowledge, performance, account, event, social, customer, fallback

Plus: reply classification, reply composition, and selector prompts.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Trigger kind → prompt family router
# ---------------------------------------------------------------------------

_TRIGGER_FAMILY: dict[str, str] = {
    # knowledge
    "research_digest": "knowledge",
    "regulation_change": "knowledge",
    "cde_opportunity": "knowledge",
    "category_research_digest_release": "knowledge",
    # performance
    "perf_dip": "performance",
    "perf_spike": "performance",
    "seasonal_perf_dip": "performance",
    # account
    "renewal_due": "account",
    "dormant_with_vera": "account",
    "gbp_unverified": "account",
    "winback_eligible": "account",
    # event
    "festival_upcoming": "event",
    "ipl_match_today": "event",
    "category_seasonal": "event",
    "weather_heatwave": "event",
    "local_event_disruption": "event",
    # social
    "milestone_reached": "social",
    "review_theme_emerged": "social",
    "curious_ask_due": "social",
    "competitor_opened": "social",
    "staff_review_highlight": "social",
    "google_algorithm_update": "social",
    "category_trend_movement": "social",
    # customer
    "recall_due": "customer",
    "appointment_tomorrow": "customer",
    "chronic_refill_due": "customer",
    "customer_lapsed_soft": "customer",
    "customer_lapsed_hard": "customer",
    "trial_followup": "customer",
    "wedding_package_followup": "customer",
    "customer_birthday_upcoming": "customer",
    # planning
    "active_planning_intent": "event",
    "supply_alert": "knowledge",
}


def get_trigger_family(trigger_kind: str) -> str:
    return _TRIGGER_FAMILY.get(trigger_kind, "fallback")


# ---------------------------------------------------------------------------
# System prompt (shared across all families — voice rules + anti-patterns)
# ---------------------------------------------------------------------------

_COMPOSE_SYSTEM = """\
You are Vera, magicpin's merchant AI assistant on WhatsApp.

VOICE RULES for {category_slug}:
- Tone: {tone}
- Allowed vocabulary: {vocab_allowed}
- Taboos (never use): {vocab_taboo}

LANGUAGE:
- Detected language for this turn: {detected_language}
- Use Hindi-English code-mix naturally when Hindi is detected (e.g., "aapke liye", "2-min ka kaam hai").

COMPULSION LEVERS — layer 2-3 per message:
1. Specificity — concrete number, date, or cited source
2. Loss aversion — "you're missing X" / "before this window closes"
3. Social proof — "3 dentists in your locality did Y"
4. Effort externalization — "I've drafted X, just say go"
5. Curiosity — "want to see who?"
6. Reciprocity — "I noticed Y, thought you'd want to know"
7. Single binary CTA — end with YES/STOP or one simple ask

ANTI-PATTERNS (judge penalizes):
- Generic offers ("Flat 30% off") when service+price is available
- Multiple CTAs in one message
- Promotional tone for clinical categories (dentists, pharmacies)
- Hallucinated data not in the provided facts
- Long preambles
- Raw field names (digest_item_id, trigger_id, etc.)
- Re-introducing yourself after the first message

CONSTRAINTS:
- Under 320 characters. Be concise.
- For customer-scoped messages: send_as = "merchant_on_behalf"
- For merchant-scoped messages: send_as = "vera"
- Only use facts provided. Do not invent numbers, names, or citations.\
"""


def build_compose_system(category: dict[str, Any], detected_language: str = "en") -> str:
    voice = category.get("voice", {})
    return _COMPOSE_SYSTEM.format(
        category_slug=category.get("slug", "unknown"),
        tone=voice.get("tone", "professional"),
        vocab_allowed=", ".join(voice.get("vocab_allowed", [])),
        vocab_taboo=", ".join(voice.get("vocab_taboo", [])),
        detected_language=detected_language,
    )


# ---------------------------------------------------------------------------
# Per-family user prompts
# ---------------------------------------------------------------------------

_FAMILY_PROMPTS: dict[str, str] = {
    "knowledge": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
{digest_section}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## FEW-SHOT EXAMPLE (9/10 merchant_fit — notice: business name, locality, cohort, offer all present)
"Dr. Meera, JIDA Oct 2026 (p.14) ki report dekhi? Dr. Meera's Dental Clinic, Lajpat Nagar ke liye relevant — 2,100-patient trial: 3-mo fluoride recall cuts caries 38% in high-risk adults. Aapke 124 patients is cohort mein hain. Abstract + Dental Cleaning @ ₹299 ke saath patient WhatsApp draft kar doon? YES/STOP"

## TASK
Write a knowledge/research message. You MUST include:
- Merchant BUSINESS NAME (not just owner name) + LOCALITY in the first 2 sentences
- The digest/research citation with source + key number
- Connection to THIS merchant's specific cohort/situation
- An offer from their active offers woven into the CTA
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "performance": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## FEW-SHOT EXAMPLE (9/10 merchant_fit)
"Dr. Bharat, Bharat Dental Care (Andheri West) ke calls last 7d mein -50% hain vs baseline 12. CTR 1.8%, Andheri peers 3.0% pe hain. Main aaj Dental Cleaning @ ₹299 ka fresh Google post draft kar deta hoon — 2 min ka kaam hai. YES?"

## TASK
Write a performance message. You MUST include:
- BUSINESS NAME + LOCALITY in the opening
- Exact metric delta + peer comparison
- Their active offer by exact title in the proposed action
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "account": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## FEW-SHOT EXAMPLE (9/10 merchant_fit)
"Anjali, Glamour Lounge Spa & Salon (Aundh, Pune) ka Pro plan 38 din se expired hai. Tabse calls -30%, aur 24 customers lapsed. Aapka Haircut @ ₹99 + FREE head massage abhi strongest offer hai — ek 7-day Aundh comeback post draft kar doon? YES/STOP"

## TASK
Write an account-state message. You MUST include:
- BUSINESS NAME + LOCALITY
- Quantified impact (days, lapsed count, metric drop)
- Active offer woven into the recovery action
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "event": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## FEW-SHOT EXAMPLE (9/10 merchant_fit)
"Suresh, aaj DC vs MI 7:30 PM Arun Jaitley Stadium. SK Pizza Junction (Sant Nagar, Delhi) ke liye perfect window — weekend match pe home-delivery orders +40% hote hain. Buy 1 Pizza Get 1 Free ka banner aaj raat ke liye draft kar doon? YES/STOP"

## TASK
Write an event/timing message. You MUST include:
- BUSINESS NAME + LOCALITY
- The specific event/timing with date/time
- Their active offer connected to the opportunity
- Urgency ("aaj", "next 2 weeks", "before window closes")
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "social": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## FEW-SHOT EXAMPLE (9/10 merchant_fit)
"Lakshmi, Studio11 Family Salon (Kapra, Hyderabad) ke 12 reviews mein stylist skill mention hua — aapki strongest demand yahi hai. Main ek Google post + WhatsApp reply template draft kar doon, Haircut @ ₹99 ke saath? Reply with service name jo aap push karna chahti ho."

## TASK
Write a social proof/review/competitor message. You MUST include:
- BUSINESS NAME + LOCALITY
- The specific social signal (review count, theme, competitor name+distance)
- Active offer in the proposed action
- Ask the merchant a question OR propose positioning
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "customer": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## CUSTOMER CONTEXT
{customer_section}

## FEW-SHOT EXAMPLE (9/10 merchant_fit — note: message is FROM merchant TO customer)
"Hi Priya, Dr. Meera's Dental Clinic (Lajpat Nagar) here 🦷 Aapki 6-month cleaning recall due hai. Wed 6 Nov 6pm ya Thu 7 Nov 5pm — Dental Cleaning @ ₹299 + complimentary fluoride. Reply 1 for Wed, 2 for Thu, ya apna time batao."

## TASK
Write a message FROM the merchant's business TO their customer:
- Greet with customer name + BUSINESS NAME + LOCALITY
- State the specific reason with concrete details (service, date, slot, price)
- Use their active offer by exact title
- Match customer language preference
- send_as = "merchant_on_behalf" (ALWAYS)

Return JSON: body, cta, send_as, suppression_key, rationale\
""",

    "fallback": """\
## TRIGGER
Kind: {trigger_kind} | Urgency: {urgency}
Payload: {payload_summary}

## WHO YOU ARE TALKING TO
{merchant_facts}

## CUSTOMER CONTEXT
{customer_section}

## FEW-SHOT EXAMPLE (9/10 merchant_fit — for a novel trigger)
"Ramesh, Apollo Health Plus Pharmacy (Malviya Nagar, Jaipur) ke liye summer alert: ORS demand +40%, sunscreen +38%. Aapka 68% repeat base ready hai — ORS + sunscreen basket ka WhatsApp note ₹499+ home delivery ke saath draft kar doon? YES/STOP"

## TASK
Write a message for this novel trigger. You MUST include:
- BUSINESS NAME + LOCALITY
- The trigger's specific "why now" fact
- Active offer or service woven into the action
- 2-3 compulsion levers
- send_as = "{send_as}"

Return JSON: body, cta, send_as, suppression_key, rationale\
""",
}


def _build_merchant_brief(facts: dict[str, Any]) -> str:
    """Convert structured merchant facts into a readable narrative brief."""
    m = facts.get("merchant", {})
    name = m.get("name", "Merchant")
    full_name = m.get("full_name", name)
    location = m.get("location", "")
    langs = m.get("languages", [])
    lang_str = "Hindi-English mix" if "hi" in langs else "English"

    perf = m.get("performance", {})
    peer = m.get("peer_comparison", {})
    offers = m.get("active_offers", [])
    best_offer = m.get("best_offer", "")
    signals = m.get("signals", [])
    review = m.get("review_theme", {})
    agg = m.get("customer_aggregate", {})
    sub = m.get("subscription", {})

    lines = [f"You are messaging {name} (address them by name), who runs {full_name}"]
    if location:
        lines[0] += f" in {location}"
    lines[0] += f". Language: {lang_str}."
    lines.append(f"Use in your message: owner name \"{name}\", business name \"{full_name}\", locality \"{location}\".")

    # Performance line
    perf_parts = []
    if perf.get("views"):
        perf_parts.append(f"{perf['views']} views")
    if perf.get("calls"):
        perf_parts.append(f"{perf['calls']} calls/month")
    if perf.get("ctr"):
        perf_parts.append(f"CTR {perf['ctr']}")
    if perf_parts:
        perf_line = ", ".join(perf_parts)
        if peer.get("ctr_vs_peer") == "below" and peer.get("peer_ctr"):
            perf_line += f" (below peer average of {peer['peer_ctr']})"
        elif peer.get("ctr_vs_peer") == "above" and peer.get("peer_ctr"):
            perf_line += f" (above peer average of {peer['peer_ctr']})"
        lines.append(f"Performance: {perf_line}.")

    # Delta
    delta = perf.get("delta_7d", {})
    delta_parts = []
    for key, val in delta.items():
        if val:
            metric = key.replace("_pct", "")
            delta_parts.append(f"{metric} {val}")
    if delta_parts:
        lines.append(f"7-day change: {', '.join(delta_parts)}.")

    # Offers
    if offers:
        lines.append(f"Active offers: {', '.join(offers[:3])}.")
    else:
        lines.append("Active offers: NONE — do not invent offers. Suggest creating one instead.")

    # Subscription
    if sub.get("status") and sub.get("days_remaining") is not None:
        lines.append(f"Subscription: {sub.get('plan', 'Pro')} plan, {sub.get('status')}, {sub['days_remaining']} days remaining.")

    # Customer aggregate
    agg_parts = []
    if agg.get("total_unique_ytd"):
        agg_parts.append(f"{agg['total_unique_ytd']} unique customers YTD")
    if agg.get("lapsed_180d_plus"):
        agg_parts.append(f"{agg['lapsed_180d_plus']} lapsed 180d+")
    if agg.get("high_risk_adult_count"):
        agg_parts.append(f"{agg['high_risk_adult_count']} high-risk adult patients")
    if agg.get("repeat_customer_pct"):
        agg_parts.append(f"{agg['repeat_customer_pct']} repeat rate")
    if agg_parts:
        lines.append(f"Customers: {', '.join(agg_parts)}.")

    # Signals
    if signals:
        lines.append(f"Signals: {', '.join(str(s) for s in signals[:4])}.")

    # Review theme
    if review.get("theme"):
        occ = review.get("occurrences", "")
        lines.append(f"Review theme: {review['theme']}" + (f" ({occ} mentions)" if occ else "") + ".")

    # Verified
    if m.get("verified") is False:
        lines.append("Google profile is NOT verified.")

    return "\n".join(lines)


def _build_trigger_brief(trigger_facts: dict[str, Any]) -> str:
    """Convert trigger payload into a readable brief."""
    payload = trigger_facts.get("payload", {})
    if not payload:
        return "No additional payload details."

    lines = []
    for key, val in payload.items():
        if val is not None and val != "" and val != []:
            label = key.replace("_", " ")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val[:5])
            elif isinstance(val, dict):
                val = ", ".join(f"{k}: {v}" for k, v in val.items() if v)
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "No additional payload details."


def _build_customer_brief(customer: dict[str, Any]) -> str:
    """Convert customer facts into a readable brief."""
    name = customer.get("name", "Customer")
    state = customer.get("state", "")
    lang = customer.get("language_pref", "")
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})

    lines = [f"Customer: {name}"]
    if state:
        lines[0] += f" (state: {state})"
    if lang:
        lines.append(f"Language preference: {lang}.")
    if rel.get("last_visit"):
        lines.append(f"Last visit: {rel['last_visit']}, {rel.get('visits_total', '?')} total visits.")
    if rel.get("services_received"):
        services = rel["services_received"]
        lines.append(f"Services: {', '.join(str(s) for s in services[-3:])}.")
    if prefs.get("preferred_slots"):
        lines.append(f"Preferred time: {prefs['preferred_slots']}.")
    return "\n".join(lines)


def build_compose_user(
    facts: dict[str, Any],
    trigger: dict[str, Any],
    templates: list[tuple[dict[str, Any], float]],
    customer: dict[str, Any] | None = None,
    mode: str = "compose",
    draft_body: str = "",
) -> str:
    trigger_facts = facts["trigger"]

    # Digest section
    digest = trigger_facts.get("digest_item", {})
    if digest:
        digest_lines = ["Digest item (cite this in your message):"]
        for k, v in digest.items():
            if v:
                digest_lines.append(f"  {k}: {v}")
        digest_section = "\n".join(digest_lines)
    else:
        digest_section = ""

    # Determine send_as
    send_as = "vera"
    if trigger_facts.get("scope") == "customer" or customer:
        send_as = "merchant_on_behalf"

    # --- POLISH mode (high-confidence deterministic draft) ---
    if mode == "polish":
        top_entry = templates[0][0] if templates else {}
        languages = facts.get("merchant", {}).get("languages", [])
        fallback_key = trigger_facts.get("suppression_key", f"{trigger_facts.get('kind', '')}:{trigger.get('id', '')}")
        return _POLISH_USER.format(
            draft_body=draft_body,
            languages=", ".join(languages) if languages else "en",
            digest_section=digest_section,
            draft_cta=top_entry.get("cta_type", "open_ended"),
            send_as=send_as,
            fallback_key=fallback_key,
        )

    # --- COMPOSE mode (per-family prompt) ---
    merchant_facts = _build_merchant_brief(facts)
    payload_summary = _build_trigger_brief(trigger_facts)

    # Template reference + mismatch warning
    template_message = ""
    template_warning = ""
    if templates:
        top_entry = templates[0][0]
        template_message = top_entry.get("sample_message", "")
        tmpl_cat = top_entry.get("category", "")
        tmpl_kind = top_entry.get("trigger_kind", "")
        our_cat = facts.get("category_slug", "")
        our_kind = trigger_facts.get("kind", "")
        mismatches = []
        if tmpl_cat and tmpl_cat not in ("generic", our_cat):
            mismatches.append(f"category ({tmpl_cat} vs yours: {our_cat})")
        if tmpl_kind and tmpl_kind != our_kind:
            mismatches.append(f"trigger kind ({tmpl_kind} vs yours: {our_kind})")
        if mismatches:
            template_warning = (
                f"\nWARNING: Reference is from a DIFFERENT {' and '.join(mismatches)}. "
                "Do NOT copy its names, offers, prices, or category details. "
                "Use ONLY the structural pattern."
            )

    if facts.get("customer"):
        customer_section = _build_customer_brief(facts["customer"])
    else:
        customer_section = "Not applicable — merchant-facing message."

    # Route to the right prompt family
    family = get_trigger_family(trigger_facts.get("kind", ""))
    template_str = _FAMILY_PROMPTS[family]

    return template_str.format(
        trigger_kind=trigger_facts.get("kind", ""),
        urgency=trigger_facts.get("urgency", ""),
        payload_summary=payload_summary,
        digest_section=digest_section,
        merchant_facts=merchant_facts,
        customer_section=customer_section,
        template_message=template_message,
        template_warning=template_warning,
        send_as=send_as,
    )


# ---------------------------------------------------------------------------
# Polish prompt (for high-confidence deterministic drafts)
# ---------------------------------------------------------------------------

_POLISH_USER = """\
## DRAFT MESSAGE (deterministic template — already has the right facts)
{draft_body}

## YOUR TASK: Polish this draft. Do NOT rewrite from scratch.
1. KEEP every number, price, date, name, and citation exactly as written.
2. Add natural Hindi-English code-mix if merchant languages include "hi".
3. Layer 1-2 extra compulsion levers: curiosity, loss aversion, effort externalization.
4. Make the CTA crisper if possible.
5. Do NOT add facts not in the draft. No hallucination.
6. Under 320 characters.

Merchant languages: {languages}
{digest_section}

## Output JSON:
- body: the polished message (preserve ALL specific facts from draft)
- cta: "{draft_cta}"
- send_as: "{send_as}"
- suppression_key: {fallback_key}
- rationale: 1 sentence\
"""


# ---------------------------------------------------------------------------
# Reply classification prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are classifying a merchant's reply in a WhatsApp conversation with Vera.

Classify the intent and decide the action.

Intent types:
- auto_reply: WhatsApp Business canned auto-response (e.g., "Thank you for contacting us", "Aapki jaankari ke liye shukriya")
- commit: merchant explicitly agrees to proceed ("ok", "yes", "go ahead", "kar do", "haan", "let's do it", "mujhe join karna hai")
- question: merchant asks a specific question about the topic
- engaged: merchant shows interest but hasn't committed
- opt_out: merchant explicitly refuses ("stop", "not interested", "don't message me", "band karo")
- off_topic: merchant asks about something outside Vera's scope (GST, tax, salary, rent, invoice)

Action rules:
- auto_reply \u2192 "wait"
- commit \u2192 "send" (IMMEDIATELY switch to action, do NOT re-qualify)
- question \u2192 "send" (answer from context)
- engaged \u2192 "send" (continue conversation)
- opt_out \u2192 "end"
- off_topic \u2192 "send" (redirect once)\
"""


def build_classify_user(message: str, history: list[dict[str, Any]]) -> str:
    history_lines: list[str] = []
    for turn in history[-6:]:
        role = turn.get("from", "unknown")
        body = str(turn.get("body", ""))[:200]
        history_lines.append(f"[{role}] {body}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior turns)"
    return f"Conversation so far:\n{history_text}\n\nLatest merchant message:\n{message}"


# ---------------------------------------------------------------------------
# Reply composition prompt
# ---------------------------------------------------------------------------

_REPLY_COMPOSE_SYSTEM = """\
You are Vera, composing a reply in an ongoing WhatsApp conversation.

RULES:
- If intent is "commit": merchant said yes. Do NOT re-qualify. State what you are doing and the immediate next step.
- If intent is "question": answer from the context below. Do not invent facts.
- If intent is "engaged": acknowledge and advance by one step.
- If intent is "off_topic": politely redirect to Vera's scope.
- Keep body under 300 characters. End with a clear next step.
- Match the detected language of the merchant's latest message.
- Do not re-introduce yourself.

DETECTED LANGUAGE: {detected_language}

MERCHANT CONTEXT:
{merchant_context}

CONVERSATION HISTORY:
{history}\
"""


def build_reply_compose_system(
    merchant_context: dict[str, Any] | None,
    history: list[dict[str, Any]],
    detected_language: str = "en",
) -> str:
    ctx = json.dumps(merchant_context or {}, indent=2, ensure_ascii=False, default=str)
    history_lines: list[str] = []
    for turn in history[-8:]:
        role = turn.get("from", "unknown")
        body = str(turn.get("body", ""))[:200]
        history_lines.append(f"[{role}] {body}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior turns)"
    return _REPLY_COMPOSE_SYSTEM.format(
        merchant_context=ctx,
        history=history_text,
        detected_language=detected_language,
    )


def build_reply_compose_user(message: str, intent: str) -> str:
    return f"Merchant's latest message: {message}\nDetected intent: {intent}\n\nCompose your reply as JSON with fields: body, cta, send_as, suppression_key, rationale."
