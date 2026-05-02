# Vera Bot v2 — Hybrid BM25 + LLM Merchant Engagement

## Approach

Dual-path architecture: every message is composed by **both** a deterministic template engine and an LLM, then a selector picks the winner.

```
Input ──┬── Path A: BM25 template (deterministic, ~0ms) ──┐
        │                 parallel                          ├── LLM Selector ── winner
        └── Path B: LLM compose (per-family prompt, ~2-3s)─┘
                                                          (timeout → Path A)
```

This gets the best of both worlds:
- **Known triggers**: deterministic templates win on specificity (exact numbers, dates, citations preserved verbatim)
- **Novel triggers** (Phase 3 injections): LLM composes from scratch and scores 2x the deterministic fallback
- **Timeout/failure**: always falls back to deterministic — zero-risk degradation

## Architecture Decisions

### Why not pure LLM?
We tested pure LLM composition vs deterministic templates side-by-side with a gpt-5.4 judge. On known triggers where the template was already good, the LLM **scored worse** because it paraphrased specific numbers ("calls down 50%" became "calls down significantly") and occasionally hallucinated facts. The deterministic template preserves every number verbatim.

### Why not pure deterministic?
The old v1 submission was fully deterministic (no LLM). It scored 30.2/50 on known triggers and 23.9/50 on novel triggers. It cannot adapt to Phase 3 context injections (new digest items, updated performance, novel trigger kinds). The LLM path handles these.

### Why dual-path + selector?
Running both paths in parallel and letting an LLM selector pick the best message gives:
- +6.8 on known triggers (selector picks deterministic when it's better)
- +15.9 on novel triggers (selector picks LLM when template is weak)
- Zero losses — the selector never picks a worse message

### BM25 threshold for routing
We measured BM25 scores across all trigger kinds:
- Known triggers: min 3.89, max 15.28, mean 9.96
- Novel triggers (simulated Phase 3): all at 2.03
- **Threshold: 3.5** — below this, skip the selector and use LLM directly (no good template exists)

### Per-family prompt templates
Instead of one generic prompt, we route triggers to 7 specialized prompt families:

| Family | Triggers | Strategy |
|---|---|---|
| knowledge | research_digest, regulation_change, cde_opportunity | Cite source, clinical tone |
| performance | perf_dip, perf_spike, seasonal_perf_dip | Lead with metric delta, peer comparison |
| account | renewal_due, dormant, gbp_unverified, winback | Account risk, quantify impact |
| event | festival, ipl_match, seasonal, weather, local_event | Timing urgency, demand shift |
| social | milestone, review_theme, curious_ask, competitor | Social proof, ask the merchant |
| customer | recall_due, appointment, refill, lapsed, trial | Formal, service+slot+price |
| fallback | any novel trigger kind | First principles from payload |

Router: simple dict lookup `trigger.kind -> family` (~40 mapped kinds, unknown -> fallback).

## Challenge Pain Points — How We Solve Them

| Pain Point | Solution |
|---|---|
| Auto-reply pollution | Tier 1: keyword detection (11 markers + Devanagari regex). Tier 2: LLM classification. Max 1 retry then END. Never burns 2-3 turns. |
| Intent-handoff failures | LLM classifies commit intent (handles Hindi "kar do", "haan", implicit yes). Prompt forbids re-qualifying after commit. |
| Generic copy | Per-family prompts enforce service+price anchoring, Hindi-English code-mix, 2-3 layered compulsion levers. |
| Low engagement frequency | LLM composes for any novel trigger kind via the fallback family. BM25 finds closest structural reference. |
| Multi-turn cadence | Store tracks sends per merchant per 24h window. Max 5 messages/merchant/day. |
| Language detection | Per-turn detection (Hindi markers + Devanagari). Sticky per conversation — once Hindi detected, stays hi-en. Passed to system prompt. |
| Knowing when to stop | Unanswered nudge counter per merchant. 3+ consecutive bot messages without merchant reply = stop nudging. |

## Evaluation Results

### Judge Simulator (gpt-5.4 judge)

```bash
BOT_URL=http://localhost:8080 LLM_MODEL=gpt-5.4 TEST_SCENARIO=full_evaluation python judge_simulator.py
```

| Dimension | Score |
|---|---|
| Avg Specificity | 7/10 |
| Avg Category Fit | 8/10 |
| Avg Merchant Fit | 7/10 |
| Avg Decision Quality | 8/10 |
| Avg Engagement | 7/10 |
| **Average** | **38.7/50 (77%)** |

| Scenario | Result |
|---|---|
| Warmup (healthz, metadata, context push) | PASS |
| Auto-reply detection | PASS — ended in **1 turn** |
| Intent transition ("Ok lets do it") | PASS — switched to action mode |
| Hostile handling ("Stop messaging me") | PASS — ended immediately |

### Models Used

| Role | Model |
|---|---|
| Bot (compose + classify + select) | gpt-5.4-mini |
| Judge (simulator) | gpt-5.4 |

For detailed comparative analysis (v1 deterministic vs v2 hybrid, A/B controlled evaluation, architecture iteration history), see [docs/evaluation-analysis.md](docs/evaluation-analysis.md).

## Tech Stack

- **Python 3.12+** with FastAPI + uvicorn (async)
- **OpenAI SDK 2.33** — Responses API with `client.responses.parse()` + Pydantic `text_format` for structured output
- **rank-bm25** — pure Python BM25 retrieval over 31-entry template corpus
- **uv** — package manager (replaces pip)

## Run Locally

### Option 1: Direct (recommended for development)

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your OpenAI API key
echo 'OPENAI_API_KEY=sk-your-key-here' > .env

# Run the server
uv run uvicorn bot:app --host 0.0.0.0 --port 8080

# Run tests
uv run pytest tests/ -v

# Run judge simulator
BOT_URL=http://localhost:8080 LLM_PROVIDER=openai \
  LLM_API_KEY=sk-your-key LLM_MODEL=gpt-4o-mini \
  TEST_SCENARIO=all uv run python judge_simulator.py

# Run A/B evaluation (controlled comparison — requires bot running on :8080)
uv run python tools/evaluate_ab.py

# Override judge model:
JUDGE_MODEL=gpt-5.4 uv run python tools/evaluate_ab.py
```

### Option 2: Docker

```bash
# Build
docker build -t vera-bot .

# Run (pass API key as secret — never bake into image)
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-your-key vera-bot

# Or with a .env file
docker run -p 8080:8080 --env-file .env vera-bot
```

### Option 3: Railway

1. Push to a Git repo
2. Connect repo in Railway dashboard
3. Add `OPENAI_API_KEY` as a Railway environment variable (Settings > Variables)
4. Railway auto-detects `Dockerfile` and `railway.json`
5. Deploys to `https://your-app.up.railway.app`
6. Set `BOT_URL=https://your-app.up.railway.app` in the judge

Railway will use `$PORT` from the environment automatically. The `railway.json` and `Dockerfile` are pre-configured.

**Important**: Never put your API key in the Dockerfile, railway.json, or any committed file. Use Railway's environment variables (dashboard or `railway variables set OPENAI_API_KEY=sk-...`).

## File Structure

```
vera_bot/
  __init__.py           # Version (2.0.0)
  main.py               # FastAPI app
  config.py             # Models, timeouts, thresholds (.env loaded via dotenv)
  routes.py             # 5 endpoints: healthz, metadata, context, tick, reply
  composer.py           # Dual-path: deterministic + LLM + selector
  reply_handler.py      # Two-tier: keyword fast-path + LLM intent classification
  fact_extractor.py     # Pure-function data extraction from context dicts
  retrieval.py          # BM25 index over templates_corpus.json (31 entries)
  llm_client.py         # Async OpenAI client (responses.parse + text_format)
  prompts.py            # 7 per-family compose prompts + classify + reply + selector
  templates_corpus.json # BM25 corpus: trigger patterns + gold examples
  validators.py         # Output shape validation (URL stripping, length cap)
  schemas.py            # Pydantic request/response models
  store.py              # Context store, suppression, cadence, nudge tracking, language detection
bot.py                  # Root shim (imports compose + app)
Dockerfile              # Production container
railway.json            # Railway deployment config
pyproject.toml          # uv project config
```

## Tradeoffs

- **Dual-path adds latency (~3-4s per message)** but stays well within the 30s budget. Parallel execution keeps it fast even with 20 actions per tick.
- **BM25 retrieval is lexical, not semantic** — works well for trigger-kind matching but might miss creative cross-trigger connections. Good enough for the corpus size (31 entries).
- **Selector adds an extra LLM call** but prevents the LLM from averaging down deterministic quality. The fallback-to-deterministic on timeout means zero downside.
- **In-memory state** is sufficient for the judge window. Production would need Redis/persistent storage.
- **No reasoning model used** — gpt-5.4-mini is fast enough for the composition task. A reasoning model could improve the selector's judgment but would add latency.
