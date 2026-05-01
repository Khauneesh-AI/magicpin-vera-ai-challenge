from __future__ import annotations

import re
from typing import Any

from vera_bot import config

MAX_BODY_CHARS = config.MAX_BODY_CHARS
def _first_name(merchant: dict[str, Any]) -> str:
    ident = merchant.get("identity", {})
    owner = ident.get("owner_first_name") or ident.get("name", "Merchant")
    owner = str(owner).replace("Dr.", "Dr.").strip()
    category = merchant.get("category_slug", "")
    if category == "dentists" and not owner.lower().startswith("dr"):
        return f"Dr. {owner}"
    return owner


def _merchant_short_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("name") or _first_name(merchant)


def _city_locality(merchant: dict[str, Any]) -> str:
    ident = merchant.get("identity", {})
    locality = ident.get("locality")
    city = ident.get("city")
    if locality and city:
        return f"{locality}, {city}"
    return locality or city or "your area"


def _active_offer(merchant: dict[str, Any], category: dict[str, Any] | None = None) -> str:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return str(offer["title"])
    if category:
        for offer in category.get("offer_catalog", []):
            if offer.get("title"):
                return str(offer["title"])
    return ""


def _peer_ctr(category: dict[str, Any]) -> float | None:
    stats = category.get("peer_stats", {})
    value = stats.get("avg_ctr")
    return float(value) if isinstance(value, (int, float)) else None


def _pct(value: Any, signed: bool = False) -> str:
    if isinstance(value, (int, float)):
        n = value * 100 if abs(value) <= 1 else value
        sign = "+" if signed and n > 0 else ""
        return f"{sign}{n:.0f}%"
    return str(value)


def _abs_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _pct(abs(value))
    return str(value).lstrip("+-")


def _pct_field(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.0f}%" if abs(value) <= 1 else f"{value:.0f}%"
    return str(value)


def _money(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"Rs {value}"


def _short_date(value: Any) -> str:
    text = str(value or "").split("T")[0]
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return _humanize(text)
    _, month, day = match.groups()
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    month_index = max(1, min(12, int(month))) - 1
    return f"{int(day)} {months[month_index]}"


def _metric_verb(metric: Any) -> str:
    return "are" if str(metric).strip().lower().endswith("s") else "is"


def _payload_facts(payload: dict[str, Any], limit: int = 2) -> list[str]:
    facts: list[str] = []
    skip = {"category", "merchant_id", "customer_id"}
    for key, value in payload.items():
        if key in skip or value in (None, "", [], {}):
            continue
        label = _humanize(key)
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, (int, float)):
            rendered = _pct(value) if "pct" in key or "delta" in key else str(value)
        elif isinstance(value, str):
            rendered = _humanize(value).split("T")[0]
        elif isinstance(value, list):
            bits = []
            for item in value[:2]:
                if isinstance(item, dict):
                    bits.append(str(item.get("label") or item.get("title") or item.get("name") or ""))
                else:
                    bits.append(_humanize(item))
            rendered = ", ".join(bit for bit in bits if bit)
        else:
            continue
        if rendered:
            facts.append(f"{label} {rendered}")
        if len(facts) >= limit:
            break
    return facts


def _top_review_theme(merchant: dict[str, Any]) -> str:
    themes = merchant.get("review_themes", [])
    if not themes:
        return ""
    theme = themes[0]
    name = _humanize(theme.get("theme", ""))
    count = theme.get("occurrences_30d")
    if name and count:
        return f"{count} reviews mention {name}"
    return name


def _matched_offer(merchant: dict[str, Any], category: dict[str, Any] | None, *terms: str) -> str:
    term_list = [term.lower() for term in terms if term]
    offers = list(merchant.get("offers", []))
    if category:
        offers.extend(category.get("offer_catalog", []))
    for offer in offers:
        title = str(offer.get("title", ""))
        if offer.get("status") not in (None, "active") or not title:
            continue
        lowered = title.lower()
        if any(term in lowered for term in term_list):
            return title
    return _active_offer(merchant, category)


def _merchant_anchor(merchant: dict[str, Any], offer: str = "") -> str:
    parts: list[str] = []
    location = _city_locality(merchant)
    if location != "your area":
        parts.append(location)
    perf = merchant.get("performance", {})
    if perf.get("views") is not None and perf.get("calls") is not None:
        parts.append(f"{perf.get('views')} views/{perf.get('calls')} calls")
    if offer:
        parts.append(offer)
    return "; ".join(parts[:3])


def _find_digest(category: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
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


def _customer_name(customer: dict[str, Any] | None) -> str:
    if not customer:
        return ""
    name = str(customer.get("identity", {}).get("name") or "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _greeting(customer: dict[str, Any] | None, salutation: str = "Hi") -> str:
    name = _customer_name(customer)
    return f"{salutation} {name}" if name else salutation


def _humanize(value: object) -> str:
    text = str(value or "").replace("_", " ").strip()
    text = re.sub(r"\b30day\b", "30-day", text, flags=re.I)
    text = re.sub(r"\b(\d+)\s+month\b", r"\1-month", text, flags=re.I)
    text = re.sub(r"^(.+?) program 30-day$", r"30-day \1 program", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text


def _hi_en(customer: dict[str, Any] | None, merchant: dict[str, Any]) -> bool:
    if customer:
        pref = str(customer.get("identity", {}).get("language_pref", "")).lower()
        return "hi" in pref
    langs = merchant.get("identity", {}).get("languages", [])
    return "hi" in langs


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text).strip()


def _clean(text: str, max_chars: int = MAX_BODY_CHARS) -> str:
    text = _strip_urls(" ".join(str(text).split()))
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "."


def _rationale(text: str) -> str:
    return _clean(text, 240)


def _result(
    body: str,
    cta: str,
    send_as: str,
    suppression_key: str,
    rationale: str,
) -> dict[str, str]:
    return {
        "body": _clean(body),
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": _rationale(rationale),
    }


def _customer_message(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any],
) -> dict[str, str]:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    greeting = _greeting(customer)
    merchant_name = _merchant_short_name(merchant)
    offer = _active_offer(merchant, category)
    suppression = trigger.get("suppression_key", f"{kind}:{customer.get('customer_id', '')}")
    hi = _hi_en(customer, merchant)

    if kind in {"recall_due", "appointment_tomorrow"}:
        slots = payload.get("available_slots") or payload.get("next_session_options") or []
        slot_bits = [s.get("label") for s in slots if isinstance(s, dict) and s.get("label")]
        slot_text = " or ".join(slot_bits[:2]) if slot_bits else ("tomorrow" if kind == "appointment_tomorrow" else "this week")
        service = _humanize(payload.get("service_due") or ("appointment" if kind == "appointment_tomorrow" else "follow-up"))
        mix = "I can hold"
        due_line = f"Your {service} is tomorrow" if kind == "appointment_tomorrow" else f"Your {service} is due"
        body = f"{greeting}, {merchant_name} here. {due_line}. {mix} a slot for {slot_text}"
        if offer:
            body += f" with {offer}"
        body += ". Reply YES to confirm, or share another time."
        return _result(
            body,
            "binary_yes_no",
            "merchant_on_behalf",
            suppression,
            "Customer-scoped reminder uses due service, available slot, merchant identity, and active offer without medical overclaim.",
        )

    if kind == "trial_followup":
        trial_date = payload.get("trial_date")
        slot = ""
        options = payload.get("next_session_options") or []
        if options and isinstance(options[0], dict):
            slot = options[0].get("label", "")
        service_word = "kids yoga" if category.get("slug") == "gyms" else "trial"
        body = (
            f"{greeting}, {merchant.get('identity', {}).get('owner_first_name', merchant_name)} from {merchant_name} here. "
            f"Thanks for trying the {service_word} session"
        )
        if trial_date:
            body += f" on {_short_date(trial_date)}"
        body += f". I can hold the next spot on {slot}" if slot else ". I can hold the next spot this week"
        body += ". Reply YES to continue, or tell me a better time."
        return _result(
            body,
            "binary_yes_no",
            "merchant_on_behalf",
            suppression,
            "Trial follow-up references the trial date/next slot and asks for a clear continuation signal.",
        )

    if kind in {"customer_lapsed_hard", "customer_lapsed_soft"}:
        days = payload.get("days_since_last_visit")
        focus = _humanize(payload.get("previous_focus") or customer.get("preferences", {}).get("training_focus") or "")
        months = payload.get("previous_membership_months")
        detail = f"It's been {days} days" if days else "Quick check-in"
        if category.get("slug") == "gyms":
            body = (
                f"{greeting}, {merchant.get('identity', {}).get('owner_first_name', merchant_name)} from {merchant_name} here. "
                f"{detail} since your last visit; no pressure. "
            )
            if months:
                body += f"You stayed {months} months, and {focus or 'your goal'} was the focus. "
            body += "I can hold an easy comeback session this week. Reply YES; no auto-charge."
        elif category.get("slug") == "salons":
            body = (
                f"{greeting}, {merchant_name} here. We have not seen you in a while. "
                f"Want me to hold your preferred slot for {offer or 'your next service'}?"
            )
        else:
            body = (
                f"{greeting}, {merchant_name} here. {detail}. "
                f"Want me to keep {offer or 'your usual service'} ready for you this week?"
            )
        return _result(
            body,
            "binary_yes_no",
            "merchant_on_behalf",
            suppression,
            "Customer winback keeps tone warm, references relationship state, and asks for one low-friction yes.",
        )

    if kind == "wedding_package_followup":
        days = payload.get("days_to_wedding")
        window = _humanize(payload.get("next_step_window_open", "bridal prep"))
        trial = _short_date(payload.get("trial_completed")) if payload.get("trial_completed") else ""
        bridal_offer = _matched_offer(merchant, category, "spa", "skin", "bridal")
        owner = merchant.get("identity", {}).get("owner_first_name", merchant_name)
        body = (
            f"{greeting}, {owner} from {merchant_name} here. "
            f"{days} days to your wedding"
        )
        if trial:
            body += f", and your trial was on {trial}"
        body += f"; this is the right window for {window}. "
        body += f"Want me to block Saturday for {bridal_offer or 'the first bridal prep session'}?"
        return _result(
            body,
            "binary_yes_no",
            "merchant_on_behalf",
            suppression,
            "Bridal follow-up uses wedding countdown, trial context, merchant owner voice, and a simple slot-hold ask.",
        )

    if kind == "chronic_refill_due":
        if category.get("slug") != "pharmacies":
            body = (
                f"{greeting}, {merchant_name} here. Quick follow-up from your last visit is due. "
                "Want us to hold a slot this week? Reply YES, or share another time."
            )
            return _result(
                body,
                "binary_yes_no",
                "merchant_on_behalf",
                suppression,
                "Non-pharmacy merchant received a refill-style trigger, so the bot avoids medicine claims and sends a safe follow-up ask.",
            )
        meds = ", ".join(payload.get("molecule_list", [])[:3]) or "your monthly medicines"
        date = str(payload.get("stock_runs_out_iso", "")).split("T")[0] or "soon"
        delivery = " Free home delivery to saved address." if payload.get("delivery_address_saved") else ""
        body = (
            f"{_greeting(customer, 'Namaste')}, {merchant_name} here. Your {meds} stock runs out on {date}. "
            f"Same dose/brand pack can be kept ready.{delivery} Reply CONFIRM to dispatch, or call us if dose changed."
        )
        return _result(
            body,
            "binary_confirm_cancel",
            "merchant_on_behalf",
            suppression,
            "Refill message is precise, respectful, includes medicines and due date, and avoids clinical advice.",
        )

    body = (
        f"{greeting}, {merchant_name} here. Quick update for you: "
        f"{str(kind).replace('_', ' ')} is due. Reply YES if you want us to help."
    )
    return _result(body, "binary_yes_no", "merchant_on_behalf", suppression, "Generic customer-scoped fallback uses trigger kind and merchant identity.")


def _planning_message(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any]) -> dict[str, str]:
    topic = trigger.get("payload", {}).get("intent_topic", "plan").replace("_", " ")
    name = _first_name(merchant)
    offer = _active_offer(merchant, category)
    cat = category.get("slug")
    if "corporate" in topic or "thali" in topic:
        body = (
            f"{name}, starter corporate thali: 10 @ Rs125, 25 @ Rs115, 50+ @ Rs105 + free delivery. "
            f"Your {offer or 'weekday thali'} is already the anchor. Want me to draft the 3-line office WhatsApp?"
        )
    elif "kids" in topic or "yoga" in topic:
        body = (
            f"{name}, draft kids yoga camp: age 7-12, 4 weeks, 3 classes/week, Rs2,499. "
            f"Use Sat morning trial first. Want me to turn this into a GBP post + parent WhatsApp?"
        )
    else:
        body = (
            f"{name}, I have a first draft for {topic}: lead with {offer or category.get('display_name', 'your offer')}, "
            f"keep one price, and ask for YES. Want me to send the exact copy?"
        )
    return _result(
        body,
        "open_ended",
        "vera",
        trigger.get("suppression_key", f"planning:{merchant.get('merchant_id')}"),
        f"Merchant already showed planning intent; response switches to concrete action for {topic}.",
    )


def _merchant_message(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any]) -> dict[str, str]:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    name = _first_name(merchant)
    cat = category.get("slug")
    perf = merchant.get("performance", {})
    aggregate = merchant.get("customer_aggregate", {})
    offer = _active_offer(merchant, category)
    suppression = trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', '')}")

    if kind == "active_planning_intent":
        return _planning_message(category, merchant, trigger)

    if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        item = _find_digest(category, trigger)
        title = item.get("title", str(kind).replace("_", " "))
        source = item.get("source", "")
        if kind == "cde_opportunity":
            credits = payload.get("credits") or item.get("credits")
            fee = payload.get("fee") or item.get("actionable", "")
            event_time = str(item.get("date") or trigger.get("expires_at") or "")
            event_hint = ""
            if event_time:
                event_hint = f" on {_short_date(event_time)}"
                if "T" in event_time and len(event_time.split("T", 1)[1]) >= 5:
                    event_hint += f" at {event_time.split('T', 1)[1][:5]}"
            body = (
                f"{name}, IDA/CDE item for this week{event_hint}: {title}. "
                f"{credits or 2} credits; {str(fee).replace('_', ' ')}. Want a calendar note + 3-line patient-facing post?"
            )
        elif "radiograph" in title.lower() or kind == "regulation_change":
            deadline = payload.get("deadline_iso", "").split("T")[0] or "Dec 15"
            body = (
                f"{name}, compliance note: {title}. Deadline {deadline}. "
                f"If you use D-speed film, audit is needed; RVG/E-speed is safer. Want a 5-point SOP checklist?"
            )
        else:
            trial = item.get("trial_n")
            segment = str(item.get("patient_segment", "")).replace("_", " ")
            cohort = aggregate.get("high_risk_adult_count")
            body = f"{name}, {source or 'this week'}: {title}."
            if trial:
                body += f" {trial}-patient signal"
            if cohort:
                body += f"; relevant to your {cohort} high-risk adults"
            elif segment:
                body += f"; relevant to {segment}"
            body += ". Want me to pull a 2-min summary + patient WhatsApp?"
        return _result(
            body,
            "open_ended",
            "vera",
            suppression,
            "External knowledge trigger uses cited digest item, category voice, and merchant/customer cohort where present.",
        )

    if kind == "supply_alert":
        batches = ", ".join(payload.get("affected_batches", [])[:3])
        molecule = payload.get("molecule", "medicine")
        chronic = aggregate.get("chronic_rx_count")
        est = max(1, round(chronic * 0.09)) if isinstance(chronic, int) else None
        body = (
            f"{name}, urgent: {molecule} recall for batches {batches} by {payload.get('manufacturer', 'manufacturer')}. "
            f"{est or 'Some'} repeat-Rx customers may need replacement. Want me to draft the customer note + pickup workflow?"
        )
        return _result(body, "open_ended", "vera", suppression, "Supply alert prioritizes batch-specific compliance and offers a complete replacement workflow.")

    if kind in {"perf_dip", "seasonal_perf_dip"}:
        metric = payload.get("metric", "calls")
        delta = payload.get("delta_pct", perf.get("delta_7d", {}).get(f"{metric}_pct"))
        peer = _peer_ctr(category)
        if kind == "seasonal_perf_dip" and payload.get("is_expected_seasonal"):
            members = aggregate.get("total_active_members")
            verb = _metric_verb(metric)
            body = (
                f"{name}, {metric} {verb} down {_abs_pct(delta)} this week, but Apr-Jun is a normal gym lull. "
                f"Skip extra ads; protect {members or 'current'} members with a summer attendance challenge. Want the draft?"
            )
        else:
            baseline = payload.get("vs_baseline")
            verb = _metric_verb(metric)
            if isinstance(delta, (int, float)) and delta > 0:
                body = f"{name}, Vera flagged a {metric} dip, but latest data shows {_pct(delta, signed=True)}. Better not panic"
            else:
                body = f"{name}, {metric} {verb} down {_abs_pct(delta)} vs {payload.get('window', '7d')}"
            if baseline:
                body += f" baseline {baseline}"
            if peer and perf.get("ctr"):
                body += f"; CTR {perf.get('ctr')*100:.1f}% vs peer {peer*100:.1f}%"
            body += f". Want me to refresh your post around {offer or 'one specific offer'} today?"
        return _result(body, "binary_yes_no", "vera", suppression, "Performance dip trigger uses metric movement and proposes one immediate recovery action.")

    if kind == "perf_spike":
        metric = payload.get("metric", "calls")
        delta = payload.get("delta_pct")
        if not isinstance(delta, (int, float)):
            delta = perf.get("delta_7d", {}).get(f"{metric}_pct")
        driver = str(payload.get("likely_driver") or "recent activity").replace("_", " ")
        baseline = payload.get("vs_baseline") or perf.get(metric)
        verb = _metric_verb(metric)
        body = f"{name}, {metric} {verb} up {_pct(delta, signed=True)}"
        if baseline:
            body += f" vs baseline {baseline}"
        body += f". Looks tied to {driver}. Want me to double down with a fresh post today?"
        return _result(body, "binary_yes_no", "vera", suppression, "Performance spike trigger captures momentum and asks for one amplification action.")

    if kind == "competitor_opened":
        comp = payload.get("competitor_name", "a new competitor")
        distance = payload.get("distance_km")
        their_offer = payload.get("their_offer")
        body = (
            f"{name}, {comp} opened {distance}km away"
            if distance
            else f"{name}, {comp} opened nearby"
        )
        if their_offer:
            body += f" with {their_offer}"
        body += f". Your counter should not be cheaper; use {offer or 'your strongest service'} + reviews. Want the draft?"
        return _result(body, "binary_yes_no", "vera", suppression, "Competitor trigger uses distance/offer and recommends positioning rather than blind discounting.")

    if kind == "curious_ask_due":
        theme = _top_review_theme(merchant)
        signal = _humanize(merchant.get("signals", [""])[0] if merchant.get("signals") else "")
        clue = theme or signal or "this week's customer demand"
        body = (
            f"{name}, quick check: {clue}. Is that still the strongest demand at {_merchant_short_name(merchant)}? "
            "Reply with one service name; I will turn it into a Google post + 4-line WhatsApp reply."
        )
        return _result(body, "open_ended", "vera", suppression, "Curious-ask trigger invites merchant input and offers to convert it into useful copy.")

    if kind in {"festival_upcoming", "ipl_match_today", "category_seasonal"}:
        if kind == "ipl_match_today":
            body = (
                f"{name}, {payload.get('match')} at {payload.get('venue')} tonight {payload.get('match_time_iso', '')[11:16]}. "
                f"Since it is not weeknight, push delivery over dine-in. Use {offer or 'your active combo'}. Want banner copy?"
            )
        elif kind == "category_seasonal":
            trends = ", ".join(str(x).replace("_", " ") for x in payload.get("trends", [])[:3])
            repeat = aggregate.get("repeat_customer_pct")
            repeat_text = f"{_pct_field(repeat)} repeat base" if repeat is not None else "active customer base"
            body = (
                f"{name}, summer shelf shift: {trends}. Your {repeat_text} is useful here. "
                "Want a WhatsApp note for ORS/sunscreen baskets?"
            )
        else:
            festival = payload.get("festival", "upcoming festival")
            days = payload.get("days_until")
            timing = f"{festival} is {days} days away" if days not in (None, "") else f"{festival} window is coming"
            body = (
                f"{name}, {timing}. "
                f"Best move: one service-price post, not a blanket discount. Want me to draft it around {offer or 'your top service'}?"
            )
        return _result(body, "open_ended", "vera", suppression, "Seasonal/event trigger links timing with a category-appropriate action and existing offer.")

    if kind in {"review_theme_emerged", "milestone_reached"}:
        if kind == "review_theme_emerged":
            theme = str(payload.get("theme", "review theme")).replace("_", " ")
            count = payload.get("occurrences_30d")
            body = (
                f"{name}, {count} reviews now mention {theme}. Quote pattern: "
                f"'{payload.get('common_quote', 'same issue')}'. Want a reply template + ops note?"
            )
        else:
            metric = str(payload.get("metric", "milestone")).replace("_", " ")
            if payload.get("value_now") is not None and payload.get("milestone_value") is not None:
                body = (
                    f"{name}, you are at {payload.get('value_now')} {metric}; {payload.get('milestone_value')} is within reach. "
                    "Want me to draft a review-request WhatsApp for recent happy customers?"
                )
            else:
                body = (
                    f"{name}, Vera flagged a {metric} milestone for {_merchant_short_name(merchant)}. "
                    f"With {perf.get('calls', 'recent')} calls this month, a review-request WhatsApp is the cleanest next step. Want the draft?"
                )
        return _result(body, "open_ended", "vera", suppression, "Review/milestone trigger turns social proof into one concrete engagement action.")

    if kind in {"renewal_due", "winback_eligible", "dormant_with_vera", "gbp_unverified"}:
        if kind == "renewal_due":
            days = payload.get("days_remaining", merchant.get("subscription", {}).get("days_remaining"))
            plan = payload.get("plan") or merchant.get("subscription", {}).get("plan") or "Pro"
            amount = _money(payload.get("renewal_amount"))
            calls_delta = perf.get("delta_7d", {}).get("calls_pct")
            peer = _peer_ctr(category)
            body = (
                f"{name}, {plan} renewal is in {days} days"
            )
            if amount:
                body += f" ({amount})"
            if calls_delta is not None:
                direction = "down" if isinstance(calls_delta, (int, float)) and calls_delta < 0 else "up"
                body += f"; calls are {direction} {_abs_pct(calls_delta)}"
            if peer and perf.get("ctr"):
                body += f", CTR {perf.get('ctr')*100:.1f}% vs peer {peer*100:.1f}%"
            body += ". Want a renewal decision note with the 3 fixes most likely to recover leads?"
        elif kind == "gbp_unverified":
            uplift = payload.get("estimated_uplift_pct")
            views = perf.get("views")
            extra = round(views * uplift) if isinstance(views, (int, float)) and isinstance(uplift, (int, float)) else None
            body = (
                f"{name}, your Google profile is still unverified in {_city_locality(merchant)}. "
                f"Verification can lift visibility by about {_pct(uplift)}"
            )
            if extra:
                body += f"; at {views} views/month, that is roughly {extra} extra profile views"
            body += ". Want the postcard/phone steps?"
        elif kind == "winback_eligible":
            days = payload.get("days_since_expiry")
            lapsed = payload.get("lapsed_customers_added_since_expiry")
            if cat == "salons":
                body = (
                    f"{name}, since expiry {days} days ago, booking calls slipped {_abs_pct(payload.get('perf_dip_pct'))} "
                    f"and {lapsed} more clients moved into lapsed. Want a 7-day Aundh comeback post + WhatsApp?"
                )
            else:
                body = (
                    f"{name}, since expiry {days} days ago, calls fell {_abs_pct(payload.get('perf_dip_pct'))} "
                    f"and {lapsed} more customers lapsed. Want a 7-day restart plan?"
                )
        else:
            days = payload.get("days_since_last_merchant_message", "many")
            last_topic = payload.get("last_topic")
            topic_note = f"{_humanize(last_topic)} note" if last_topic else "note"
            anchor = _merchant_anchor(merchant, _active_offer(merchant, None))
            task = "service-price comeback post" if cat == "salons" else "profile recovery task"
            body = (
                f"{name}, it has been {days} days since our last {topic_note}. "
                f"For {_merchant_short_name(merchant)}"
            )
            if anchor:
                body += f" ({anchor})"
            body += f", the clean next step is one {task}. Want the 2-line version?"
        return _result(body, "open_ended", "vera", suppression, "Account-state trigger uses current account risk and asks for a low-effort next step.")

    metric = payload.get("metric_or_topic") or kind.replace("_", " ")
    facts = _payload_facts(payload, limit=2)
    anchor = _merchant_anchor(merchant, offer)
    body = (
        f"{name}, quick signal for {_merchant_short_name(merchant)}"
    )
    if anchor:
        body += f" in {anchor}"
    body += f": {_humanize(metric)}"
    if facts:
        body += f" ({'; '.join(facts)})"
    body += ". Want me to turn this into one customer-facing post or WhatsApp draft?"
    return _result(body, "open_ended", "vera", suppression, "Fallback chooses a concrete merchant metric plus trigger topic and asks for one draft action.")

def missing_customer_message(merchant: dict[str, Any], trigger: dict[str, Any]) -> dict[str, str]:
    return _result(
        f"{_first_name(merchant)}, customer trigger {trigger.get('kind')} is active, but customer context is missing. Want me to wait?",
        "none",
        "vera",
        trigger.get("suppression_key", f"missing_customer:{trigger.get('id', '')}"),
        "Avoids fabricating customer facts when customer context was not supplied.",
    )


def customer_message(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any],
) -> dict[str, str]:
    return _customer_message(category, merchant, trigger, customer)


def merchant_message(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any]) -> dict[str, str]:
    return _merchant_message(category, merchant, trigger)
