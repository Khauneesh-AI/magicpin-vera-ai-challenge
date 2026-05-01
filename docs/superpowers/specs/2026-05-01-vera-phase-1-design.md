# Vera AI Challenge Phase 1 Design

Date: 2026-05-01

## Purpose

Build a reliable Magicpin Vera challenge submission that satisfies the required composition contract, HTTP API, state handling, and submission artifacts. Phase 1 is deterministic-only: no Gemini, OpenAI, or other live LLM calls in the submitted API.

The design optimizes for judge compatibility, fast responses, clear code, and low operational risk. LLMs may be used later as offline assistants to improve templates, but their output must be baked into deterministic code before submission.

## Approved Approach

Use Approach C:

1. Freeze the current `bot.py` as a prototype/reference.
2. Build validation tools first so regressions are visible.
3. Rebuild cleanly into a flat `vera_bot/` package.
4. Generate and validate `submission.jsonl`.
5. Smoke-test the HTTP API locally before deployment.

## Scope

Phase 1 includes:

- Pure deterministic `compose(category, merchant, trigger, customer=None) -> dict`.
- `submission.jsonl` generation for the 30 canonical test pairs.
- `README.md` for the final submission.
- Required HTTP endpoints:
  - `GET /v1/healthz`
  - `GET /v1/metadata`
  - `POST /v1/context`
  - `POST /v1/tick`
  - `POST /v1/reply`
- In-memory context and conversation state.
- Suppression and anti-repeat behavior.
- Basic multi-turn reply handling.
- Deterministic validation and smoke-test tools.

Phase 1 excludes:

- Live LLM calls.
- `model_handler.py`.
- `model_endpoints.py`.
- Provider catalogs or model health probes.
- Any self-evaluator service.
- Database persistence.
- URLs in generated messages.

## File Layout

```text
bot.py
requirements.txt
README.md
submission.jsonl

vera_bot/
  __init__.py
  main.py
  config.py
  schemas.py
  store.py
  composer.py
  trigger_handlers.py
  reply_handler.py
  validators.py
  routes.py

tools/
  generate_submission.py
  validate_submission.py
  smoke_endpoints.py
```

## Architecture

`bot.py` is a public shim for the judge:

```python
from vera_bot.main import app
from vera_bot.composer import compose

__all__ = ["app", "compose"]
```

The package has one clear boundary:

```python
compose(category, merchant, trigger, customer=None) -> dict
```

`compose()` is pure. It does not read from the store, call external services, use the clock, use randomness, or mutate global state. This keeps JSONL generation and deterministic replay checks simple.

Stateful behavior lives in the API layer:

- `/v1/context` stores context by `(scope, context_id)` with versions.
- `/v1/tick` loads relevant contexts, calls `compose()`, applies suppression, applies anti-repeat, and returns up to 20 actions.
- `/v1/reply` uses conversation history and reply rules to decide `send`, `wait`, or `end`.

## Data Flow

Submission generation:

1. `tools/generate_submission.py` reads `expanded/test_pairs.json`.
2. It loads category, merchant, trigger, and optional customer context.
3. It calls pure `compose()`.
4. It writes exactly 30 JSONL rows.
5. `tools/validate_submission.py` checks shape, determinism, no URLs, no `None`, and length limits.

Judge API flow:

1. Judge calls `/v1/healthz` and `/v1/metadata`.
2. Judge pushes category, merchant, customer, and trigger context through `/v1/context`.
3. Judge calls `/v1/tick` with active trigger IDs.
4. Bot sends zero or more proactive actions, max 20.
5. Judge sends merchant/customer replies to `/v1/reply`.
6. Bot continues, waits, or ends the conversation.

## Context Versioning

`/v1/context` is idempotent on `(scope, context_id, version)`.

Behavior:

- Newer version: store and return `{accepted: true, ack_id: ...}`.
- Same version: keep stored context and return `{accepted: true, ack_id: ...}`.
- Older version: do not overwrite; return `{accepted: false, reason: "stale_version", current_version: n}`.

The challenge sample returns a normal JSON body for stale context, so Phase 1 will use HTTP 200 with `accepted: false` rather than HTTP 409. This favors compatibility with the sample harness.

## Message Composition

`trigger_handlers.py` owns deterministic handler functions by trigger family:

- research and education
- compliance and supply alerts
- performance dips and spikes
- competitor openings
- festivals and seasonal moments
- review themes and milestones
- renewal, dormant, winback, and account-state triggers
- planning-intent triggers
- customer recall and appointment reminders
- lapsed customer and trial follow-up
- pharmacy refill reminders
- safe fallback handlers

Each handler returns:

```python
{
  "body": "...",
  "cta": "...",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "...",
  "rationale": "..."
}
```

The message must be specific to the available context and must not invent facts. If required context is missing, the handler should return a conservative fallback instead of fabricating.

## Validation Rules

`validators.py` applies deterministic cleanup:

- Strip all URLs in Phase 1.
- Remove accidental `None` leaks.
- Trim body length to the configured cap.
- Enforce a single clear CTA.
- Normalize whitespace.
- Keep `send_as` within accepted values.
- Ensure required output keys exist.

Anti-repeat is applied in the API flow, not in pure `compose()`, because it depends on conversation history and previous sends.

## Store

`store.py` keeps in-memory state:

```python
contexts: dict[tuple[str, str], StoredContext]
conversations: dict[str, list[ConversationTurn]]
sent_suppressions: set[tuple[str, str]]
recent_bodies: dict[str, deque[str]]
auto_reply_hashes: dict[str, list[str]]
```

Suppression keys are scoped by merchant:

```python
(merchant_id, suppression_key)
```

This avoids incorrectly suppressing category-level triggers for other merchants.

## Tick Behavior

`/v1/tick` must:

- Return within 30 seconds.
- Iterate over all `available_triggers`.
- Skip missing or already-suppressed triggers.
- Stop only after 20 valid actions have been collected.
- Return `{"actions": []}` when there is nothing useful to say.

It must not slice `available_triggers` before filtering, because the first 20 IDs may be missing, stale, or suppressed while later IDs are valid.

## Reply Behavior

`reply_handler.py` uses first-match-wins detection:

1. Opt-out or hostile message: end gracefully.
2. Off-topic request: politely redirect once.
3. Auto-reply: detect English, Hindi, romanized Hindi, and repeated identical merchant turns.
4. Explicit commit: switch to action mode immediately.
5. Ambiguous but relevant reply: send a short context-aware follow-up.
6. Unclear or low-signal reply: wait rather than spam.

The reply handler should reuse deterministic composition patterns where practical, but it must not call an LLM.

## Testing Gates

Before deployment:

- Python import/syntax checks pass.
- `tools/generate_submission.py` creates exactly 30 rows.
- `tools/validate_submission.py` exits 0.
- Deterministic replay passes byte-identically.
- No body contains `None`.
- No body contains a URL.
- All required JSONL keys are present.
- All five HTTP endpoints return expected shapes locally.
- `/v1/tick` returns max 20 actions.
- Re-sending the same trigger suppresses the duplicate for the same merchant.
- Stale `/v1/context` does not overwrite current context.
- Hindi/romanized Hindi auto-reply is detected.
- Explicit commit replies do not trigger another qualifying question.

## Deployment Assumptions

The final API should run with:

```text
uvicorn bot:app --host 0.0.0.0 --port $PORT
```

Runtime dependencies stay lean:

- fastapi
- uvicorn
- pydantic

No API keys are required for Phase 1.

## Risks And Mitigations

Risk: Deterministic messages feel less natural than LLM messages.
Mitigation: Improve templates trigger-by-trigger and keep outputs specific to context.

Risk: Refactor breaks behavior from the prototype.
Mitigation: Build validation first and compare generated submission output.

Risk: Judge pushes unseen post-submission context.
Mitigation: Store all context generically by scope/id and compose from whatever is currently stored.

Risk: Bot spams repeated messages.
Mitigation: Merchant-scoped suppression plus recent body hashes.

Risk: Ambiguous replies become generic.
Mitigation: Reply handler uses merchant/conversation context and conservative follow-ups.

## Open Decisions

- Final team name, team members, and contact email for `/v1/metadata`.
- Deployment host.
- Whether to initialize git in this workspace before implementation.

## Spec Self-Review

- No live LLM path remains in Phase 1.
- No external reference-repo dependency is required.
- `compose()` is pure and deterministic.
- Stateful behavior is outside `compose()`.
- URL policy is explicit: strip all URLs in Phase 1.
- `/v1/context` stale behavior follows the challenge sample response shape.
- `/v1/tick` avoids the pre-filter slicing bug.
- The scope is small enough for one implementation plan.
