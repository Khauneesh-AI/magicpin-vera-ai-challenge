# Vera Bot Agentic Redesign — Design Spec

**Date:** 2026-05-02
**Author:** Claude + Khauneesh
**Status:** Approved for implementation

---

## 1. Problem Statement

The current Vera bot submission is a deterministic Python rule engine — no LLM, no retrieval, no adaptability. Every message is an f-string template dispatched by `trigger.kind`. This approach:

- Cannot adapt to Phase 3 context injections (new digest items, updated perf, novel triggers)
- Cannot produce Hindi-English code-mix naturally
- Cannot classify intent beyond keyword matching (fails intent-handoff scenarios)
- Produces formulaic copy that scores low on engagement compulsion
- Forfeits the adaptation bonus entirely (~+5 per dimension)

The challenge brief explicitly recommends an LLM-based composer with routing, validation, and multi-turn state.

## 2. Design Goals

1. Solve all 4 pain points: auto-reply pollution, intent-handoff failures, generic copy, low engagement frequency
2. Adapt to novel context injections (Phase 3) without code changes
3. Stay within the 30-second per-call hard limit with 20 actions per tick
4. Preserve what works: store, schemas, validators, fact extraction helpers
5. Use the existing rule templates as retrieval grounding (not waste the work)

## 3. Architecture Overview

```
                         +----------------------+
                         |  FastAPI (/v1/...)    |
                         |  routes.py            |
                         +----------+-----------+
                                    |
              +---------------------+---------------------+
              |                     |                      |
         /v1/tick              /v1/reply             /v1/context
              |                     |                      |
              v                     v                      v
    +-------------------+  +-----------------+        store.py
    |  tick_pipeline     |  | reply_pipeline  |       (unchanged)
    |                    |  |                 |
    | 1. fact_extract    |  | 1. keyword      |
    | 2. bm25_retrieve   |  |    pre-check    |
    | 3. llm_compose     |  | 2. llm_classify |
    |    (GPT-5.4-nano)  |  |    (GPT-5.4-    |
    | 4. validate        |  |     mini)       |
    +-------------------+  | 3. route        |
                            +-----------------+
                                    |
                    +---------------+----------+
                 auto_reply      commit     engaged
                  end/wait       action     llm_reply
                                 (nano)     (mini)
```

### Approach: Hybrid — Rules as Retrieval + LLM as Composer

The existing trigger_handlers.py has ~75 (trigger_kind x category) template patterns with good data extraction logic. Rather than discard or replace them, we:

1. Extract the data-extraction helpers into `fact_extractor.py` (reusable pure functions)
2. Convert the template patterns into a BM25-indexed corpus (`templates_corpus.json`)
3. At runtime: extract facts deterministically, retrieve the closest template(s) via BM25, feed both into an LLM that composes the final message
4. Validate output deterministically (shape, length, no hallucinated URLs)

The LLM never starts from zero — it always has retrieved context as grounding.

### Why Not Full Agentic (Tool Calling)?

Tool calling adds 2-4 seconds of round-trips per message for zero benefit. The data lookup is fully determined by the trigger (`trigger.merchant_id` -> merchant, `merchant.category_slug` -> category, `trigger.customer_id` -> customer). There is no discovery or search needed. The LLM's job is composition, not retrieval.

## 4. File Structure

```
vera_bot/
  __init__.py              # unchanged
  main.py                  # unchanged
  config.py                # EXTENDED — add OpenAI models, timeouts
  schemas.py               # unchanged
  store.py                 # unchanged
  validators.py            # unchanged
  routes.py                # MODIFIED — async tick with parallel LLM calls
  composer.py              # REWRITTEN — orchestrator: extract -> retrieve -> llm -> validate
  reply_handler.py         # REWRITTEN — keyword pre-filter + LLM intent classification
  fact_extractor.py        # NEW — refactored data extraction helpers from trigger_handlers
  retrieval.py             # NEW — BM25 index over template corpus
  llm_client.py            # NEW — async OpenAI client, prompt building, structured output
  prompts.py               # NEW — system prompts per category voice, composition instructions
  templates_corpus.json    # NEW — indexed template patterns for BM25
  trigger_handlers.py      # RETIRED as code — content moved to templates_corpus.json
```

## 5. Component Design

### 5.1 Fact Extractor (`fact_extractor.py`)

Refactored from the existing trigger_handlers.py helper functions. Pure functions with no LLM dependency:

- `extract_merchant_facts(merchant, category)` -> dict with name, location, perf, offers, signals
- `extract_trigger_facts(trigger, category)` -> dict with kind, urgency, key payload fields
- `extract_customer_facts(customer)` -> dict with name, state, relationship, preferences
- `find_digest_item(category, trigger)` -> matching digest entry
- `get_peer_comparison(merchant, category)` -> CTR/rating vs peer stats

These are the same helpers currently in trigger_handlers.py (`_first_name`, `_active_offer`, `_peer_ctr`, `_city_locality`, `_find_digest`, etc.) reorganized to return structured dicts instead of building f-strings.

### 5.2 BM25 Retrieval (`retrieval.py`)

**Corpus format** (`templates_corpus.json`):

```json
[
  {
    "id": "perf_dip__dentists",
    "trigger_kind": "perf_dip",
    "category": "dentists",
    "scope": "merchant",
    "facts_used": ["metric", "delta_pct", "peer_ctr", "active_offer"],
    "cta_type": "binary_yes_no",
    "send_as": "vera",
    "sample_message": "Dr. Meera, calls are down 50% vs 7d baseline 12; CTR 1.8% vs peer 3.0%. Want me to refresh your post around Dental Cleaning @ ₹299 today?",
    "compulsion_levers": ["loss_aversion", "effort_externalization", "specificity"]
  }
]
```

**Index:** ~85 entries (75 from existing rules + 10 gold examples from challenge brief Appendix A/B and case studies).

**Query construction:**

```python
query = f"{trigger['kind']} {category['slug']} {' '.join(list(trigger.get('payload', {}).keys())[:5])}"
```

**Library:** `rank-bm25` (pure Python, zero external deps). Index built once at startup. Lookup is microseconds.

**Output:** Top-2 matching templates passed to LLM as reference examples.

### 5.3 LLM Client (`llm_client.py`)

**SDK:** OpenAI Python SDK >= 2.33.0, using the Responses API with Pydantic structured output.

**Client:** `AsyncOpenAI` for parallel tick calls via `asyncio.gather`.

**Composition call (tick pipeline):**

```python
from openai import AsyncOpenAI
from pydantic import BaseModel, Literal

class ComposedMessage(BaseModel):
    body: str
    cta: Literal["binary_yes_no", "open_ended", "none", "binary_confirm_cancel"]
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: str
    rationale: str

client = AsyncOpenAI()

response = await client.responses.parse(
    model="gpt-5.4-nano",
    text_format=ComposedMessage,
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.3,
    max_output_tokens=500,
)

message = response.output_parsed  # typed ComposedMessage
```

**Reply classification call:**

```python
class ReplyClassification(BaseModel):
    intent: Literal["auto_reply", "commit", "question", "engaged", "opt_out", "off_topic"]
    action: Literal["send", "wait", "end"]

response = await client.responses.parse(
    model="gpt-5.4-mini",
    text_format=ReplyClassification,
    input=[...],
    temperature=0.0,
)
```

**Reply composition call (when action=send):**

Uses GPT-5.4-mini with conversation history + merchant context to compose a contextual response. Same `ComposedMessage` output schema but with `send_as` always matching the original conversation's attribution.

**Error handling / fallback:**

If the OpenAI call fails or times out (>8s for composition, >5s for classification):
- Tick: fall back to the BM25 top-1 match sample_message directly (the existing deterministic template)
- Reply: fall back to the existing keyword-based decision logic

The deterministic rules become the safety net, not the primary path.

### 5.4 Prompts (`prompts.py`)

**System prompt for composition** (parameterized by category):

```
You are Vera, magicpin's merchant AI assistant on WhatsApp.

VOICE RULES for {category_slug}:
- Tone: {voice.tone}
- Allowed vocabulary: {voice.vocab_allowed}
- Taboos (never use): {voice.taboos}

LANGUAGE:
- When merchant languages include "hi": use Hindi-English code-mix naturally
- Match the merchant's language preference
- Hindi-English mix example: "Aapka Google profile abhi 62.5% complete hai"

COMPULSION LEVERS (use 1-2 per message):
1. Specificity — anchor on a concrete number, date, or cited source
2. Loss aversion — "you're missing X" / "before this window closes"
3. Social proof — "3 dentists in your locality did Y this month"
4. Effort externalization — "I've drafted X, just say go" / "2-min setup"
5. Curiosity — "want to see who?" / "want the full list?"
6. Reciprocity — "I noticed Y, thought you'd want to know"
7. Single binary CTA — end with YES/STOP or one simple ask

ANTI-PATTERNS (judge will penalize):
- Generic offers ("Flat 30% off") when service+price is available
- Multiple CTAs in one message
- Promotional tone for clinical categories (dentists, pharmacies)
- Hallucinated data not present in the provided facts
- Long preambles ("I hope you're doing well...")
- Re-introducing yourself after the first message
```

**User prompt for composition** (per message):

```
## Merchant Facts
{extracted_facts_json}

## Trigger
Kind: {trigger_kind}
Urgency: {urgency}
Why now: {trigger_payload_summary}

## Customer (if applicable)
{customer_facts_json or "Not applicable — merchant-facing message"}

## Reference Examples (adapt structure, don't copy verbatim)
Example 1: {bm25_match_1.sample_message}
Example 2: {bm25_match_2.sample_message}

## Output
Return a JSON object with these exact fields:
- body: the WhatsApp message (concise, anchor on one verifiable fact, end with CTA)
- cta: "binary_yes_no" | "open_ended" | "none" | "binary_confirm_cancel"
- send_as: "vera" (merchant-facing) | "merchant_on_behalf" (customer-facing)
- suppression_key: dedup key for this message type
- rationale: 1 sentence — why this message, what it should achieve
```

**System prompt for reply classification:**

```
You are classifying a merchant's reply in a WhatsApp conversation with Vera.

Classify the intent and decide the action.

Intent types:
- auto_reply: WhatsApp Business canned auto-response (e.g., "Thank you for contacting us", "Aapki jaankari ke liye shukriya")
- commit: merchant explicitly agrees to proceed ("ok", "yes", "go ahead", "kar do", "haan", "let's do it", "mujhe join karna hai")
- question: merchant asks a specific question about the topic
- engaged: merchant shows interest but hasn't committed
- opt_out: merchant explicitly refuses ("stop", "not interested", "don't message me")
- off_topic: merchant asks about something outside Vera's scope (GST, tax, salary, etc.)

Action rules:
- auto_reply -> "wait" (first time) or "end" (repeated)
- commit -> "send" (IMMEDIATELY switch to action, do NOT re-qualify)
- question -> "send" (answer from context)
- engaged -> "send" (continue conversation)
- opt_out -> "end"
- off_topic -> "send" (redirect once, then end)
```

### 5.5 Composer (`composer.py` — rewritten)

Orchestrator that ties the pipeline together:

```python
async def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict[str, str]:
    # 1. Extract facts (deterministic, ~0ms)
    facts = extract_all_facts(category, merchant, trigger, customer)

    # 2. Retrieve similar templates (BM25, ~0ms)
    templates = retrieval_index.query(trigger, category, top_n=2)

    # 3. Build prompt
    system_prompt = build_system_prompt(category)
    user_prompt = build_user_prompt(facts, trigger, templates, customer)

    # 4. LLM compose (GPT-5.4-nano, ~1-3s)
    try:
        message = await llm_client.compose(system_prompt, user_prompt)
    except (TimeoutError, OpenAIError):
        # Fallback: use BM25 top-1 template directly
        message = fallback_from_template(templates[0], facts)

    # 5. Validate (deterministic, ~0ms)
    return finalize_message(message, fallback_key=suppression_key)
```

### 5.6 Reply Handler (`reply_handler.py` — rewritten)

Two-tier intent classification:

```python
async def handle_reply(
    conversation_id: str,
    message: str,
    history: list[dict],
    merchant_id: str | None = None,
    merchant_context: dict | None = None,
    trigger_context: dict | None = None,
    auto_reply_count: int = 0,
) -> dict[str, Any]:

    # Tier 1: Deterministic fast-path (keyword check, ~0ms)
    if auto_reply_count >= 3 or repeated_in_history(message, history):
        return {"action": "end", "rationale": "Auto-reply loop; exiting."}
    if is_exact_opt_out(message):  # only exact matches like "stop", "unsubscribe"
        return {"action": "end", "rationale": "Explicit opt-out."}

    # Tier 2: LLM classification (GPT-5.4-mini, ~1-2s)
    try:
        classification = await llm_client.classify_reply(message, history)
    except (TimeoutError, OpenAIError):
        # Fallback: keyword-based classification (existing logic)
        return keyword_fallback_classify(message, history)

    # Route based on intent
    if classification.intent == "auto_reply":
        if auto_reply_count == 0:
            return {"action": "wait", "rationale": "First auto-reply; waiting."}
        return {"action": "end", "rationale": "Repeated auto-reply; exiting."}

    if classification.intent == "commit":
        # Compose action response — no re-qualifying
        response = await llm_client.compose_reply(
            message, history, merchant_context, "commit"
        )
        return {"action": "send", "body": response.body, "cta": response.cta,
                "rationale": "Commit detected; switching to action."}

    if classification.intent == "question":
        response = await llm_client.compose_reply(
            message, history, merchant_context, "question"
        )
        return {"action": "send", "body": response.body, "cta": response.cta,
                "rationale": "Answering merchant question from context."}

    if classification.intent == "opt_out":
        return {"action": "end", "rationale": "Opt-out detected by LLM."}

    if classification.intent == "off_topic":
        return {"action": "send",
                "body": "I can help with your Google profile and customer outreach here. Want to get back to that?",
                "cta": "binary_yes_no",
                "rationale": "Off-topic redirect."}

    # engaged or uncertain — continue
    response = await llm_client.compose_reply(
        message, history, merchant_context, "engaged"
    )
    return {"action": "send", "body": response.body, "cta": response.cta,
            "rationale": "Engaged merchant; continuing conversation."}
```

### 5.7 Routes (`routes.py` — modified)

Key change: tick handler fires all LLM calls in parallel using `asyncio.gather`:

```python
@router.post("/tick")
async def tick(body: TickBody) -> dict[str, Any]:
    # Prepare all composition tasks
    tasks = []
    for trigger_id in body.available_triggers[:config.MAX_ACTIONS_PER_TICK]:
        trigger = store.get_payload("trigger", trigger_id)
        if not trigger:
            continue
        merchant = store.get_payload("merchant", trigger.get("merchant_id"))
        category = store.get_payload("category", merchant.get("category_slug")) if merchant else None
        if not (merchant and category):
            continue
        customer = None
        if trigger.get("customer_id"):
            customer = store.get_payload("customer", trigger["customer_id"])
        suppression_key = trigger.get("suppression_key", trigger_id)
        if store.was_sent(merchant.get("merchant_id", ""), suppression_key):
            continue
        tasks.append((category, merchant, trigger, customer))

    # Fire all LLM compositions in parallel
    results = await asyncio.gather(
        *[compose(cat, mer, trg, cust) for cat, mer, trg, cust in tasks],
        return_exceptions=True,
    )

    # Collect successful actions
    actions = []
    for (cat, mer, trg, cust), result in zip(tasks, results):
        if isinstance(result, Exception):
            continue
        if store.seen_recent_body(mer.get("merchant_id", ""), result["body"]):
            continue
        action = action_from_message(mer, trg, result, cust)
        actions.append(action)
        store.mark_sent(mer.get("merchant_id", ""), result["suppression_key"])
        store.remember_body(mer.get("merchant_id", ""), result["body"])

    return {"actions": actions}
```

Reply handler similarly becomes async.

## 6. Pain Point Solutions

### 6.1 Auto-Reply Pollution (Pain Point 1)

**Current:** Keyword matching detects, but burns 2-3 turns with generic responses.

**New:** Two-tier detection. Tier 1 (keywords) catches obvious cases instantly. Tier 2 (LLM) catches Hindi auto-replies and ambiguous cases. Max 1 retry with a context-specific "Is [owner_name] available? I have a quick update about [specific thing]." then END. Never burn more than 1 turn.

### 6.2 Intent-Handoff Failures (Pain Point 2)

**Current:** Keyword match "ok"/"yes" returns canned "Done. I will keep it short..."

**New:** LLM classifies commit intent (handles Hindi "mujhe join karna hai", implicit yes, "let's do it" after qualifying). On commit detection, LLM composes an action response using merchant context. The prompt explicitly forbids re-qualifying: "The merchant said yes. State what you're doing and the immediate next step."

### 6.3 Generic Copy (Pain Point 3)

**Current:** f-string templates produce correct but formulaic messages.

**New:** LLM composes with BM25 template as structural reference. Adds Hindi-English code-mix via voice rules in the system prompt. Anchors on service+price from extracted facts. Uses compulsion levers (curiosity, social proof, loss aversion) naturally.

### 6.4 Low Engagement Frequency (Pain Point 4)

**Current:** Only fires on the ~15 hardcoded trigger kinds. Novel triggers hit a weak generic fallback.

**New:** BM25 finds the closest template for any trigger kind. LLM adapts it to the actual trigger payload. Can compose curious-ask, knowledge-driven, and trend-based messages even for trigger kinds never seen during development. This directly addresses Phase 3 novel trigger injection.

## 7. Timing Budget

### Tick (worst case: 20 triggers)

| Step | Per-message | Total (parallel) |
|---|---|---|
| Fact extraction | ~1ms | 20ms |
| BM25 retrieval | ~0.1ms | 2ms |
| LLM composition (GPT-5.4-nano) | ~1-3s | ~3s (parallel) |
| Validation | ~1ms | 20ms |
| **Total** | | **~3-4s** |

### Reply (single call)

| Step | Time |
|---|---|
| Keyword pre-check | ~0ms |
| LLM intent classification (GPT-5.4-mini) | ~1-2s |
| LLM response composition (if needed) | ~2-3s |
| **Total** | **~3-5s** |

Both well within the 30-second budget.

## 8. Fallback Strategy

Every LLM call has a deterministic fallback:

| Call | Timeout | Fallback |
|---|---|---|
| Tick composition | 8s | BM25 top-1 sample_message with fact substitution |
| Reply classification | 5s | Keyword-based classification (existing logic) |
| Reply composition | 5s | Canned responses per intent type |

The system degrades gracefully to the current deterministic behavior if OpenAI is down.

## 9. Configuration

```python
# config.py
import os

TEAM_NAME = "DakshTheCoder"
TEAM_MEMBERS: list[str] = ["Daksh Malhotra"]
CONTACT_EMAIL = "dakshmalhotra_23ep033@dtu.ac.in"
APPROACH = "hybrid-bm25-retrieval-plus-llm-composer"
MODEL_NAME = "gpt-5.4-nano + gpt-5.4-mini"

# LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COMPOSE_MODEL = "gpt-5.4-nano"
CLASSIFY_MODEL = "gpt-5.4-mini"
COMPOSE_TEMPERATURE = 0.3
CLASSIFY_TEMPERATURE = 0.0
COMPOSE_TIMEOUT = 8.0
CLASSIFY_TIMEOUT = 5.0
COMPOSE_MAX_TOKENS = 500
CLASSIFY_MAX_TOKENS = 100

# Existing (unchanged)
MAX_BODY_CHARS = 320
MAX_ACTIONS_PER_TICK = 20
ANTI_REPEAT_WINDOW = 5
ALLOWED_SEND_AS = {"vera", "merchant_on_behalf"}
```

## 10. Dependencies

Managed via `uv` with `pyproject.toml`:

```toml
[project]
name = "vera-bot"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.0",
    "openai>=2.33",
    "rank-bm25>=0.2.3",
    "httpx>=0.28",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

## 11. Testing Strategy

- **Unit tests:** fact extractor functions, BM25 index query, prompt building, validator
- **Integration tests:** full compose pipeline with mocked OpenAI responses
- **Contract tests:** all 30 test pairs produce valid schema output (existing tests, adapted)
- **Fallback tests:** verify deterministic fallback activates on LLM timeout/error
- **Reply handler tests:** auto-reply detection, commit routing, opt-out handling (existing + LLM-based)

## 12. What's Preserved From Current Solution

| Component | Status |
|---|---|
| `store.py` — context storage, suppression, anti-repeat | Unchanged |
| `schemas.py` — Pydantic request/response models | Unchanged |
| `validators.py` — output shape validation | Unchanged |
| `main.py` — FastAPI app setup | Unchanged |
| `__init__.py` — version | Updated to 2.0.0 |
| Data extraction helpers | Moved to `fact_extractor.py` |
| Template patterns | Converted to `templates_corpus.json` |
| Reply keyword detection | Kept as Tier 1 fast-path |
