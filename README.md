# Vera Challenge Submission — DakshTheCoder

## Approach

Deterministic Python composer for Magicpin Vera merchant engagement. The bot uses category, merchant, trigger, and optional customer context to produce specific WhatsApp-style messages without live LLM calls.

The public contract is:

- `compose(category, merchant, trigger, customer=None) -> dict`
- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

The root `bot.py` is a shim over the flat `vera_bot/` package. Message generation lives in `vera_bot.composer` and `vera_bot.trigger_handlers`; API state lives in `vera_bot.store` and `vera_bot.routes`.

## Tradeoffs

- Deterministic rules are stable and fast, but less flexible than a live LLM.
- Missing facts produce conservative fallbacks rather than invented details.
- URLs are stripped in Phase 1 to avoid accidental fabrication.
- In-memory state is enough for the judge window, but a production bot would persist context and conversations.

## Run Locally

```powershell
python tools\generate_submission.py
python tools\validate_submission.py
uvicorn bot:app --host 0.0.0.0 --port 8000
python tools\smoke_endpoints.py http://127.0.0.1:8000
```
