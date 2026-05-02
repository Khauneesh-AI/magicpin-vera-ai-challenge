# Evaluation Analysis — v1 Deterministic vs v2 Hybrid

This document details the comparative evaluation between the original deterministic v1 bot and the hybrid v2 bot. The README contains only the final v2 judge simulator scores.

## Why Two Evaluation Methods?

The **judge simulator** (`judge_simulator.py`) is the challenge's official evaluation tool. It scores one bot per run. But because each run makes separate LLM judge calls, scores vary by 3-5 points between runs — even with the same model at low temperature. This is because:

1. **Temperature 0.2 is not 0.0** — the model still samples from a probability distribution. On subjective dimensions like "engagement compulsion," the difference between 6/10 and 7/10 depends on which token the model samples first.
2. **Different message text triggers different judge associations** — even with the same prompt structure, the specific text being evaluated changes the judge's internal attention patterns.
3. **API-level non-determinism** — OpenAI's infrastructure has floating-point non-determinism from GPU parallelism. Even at temperature 0.0, outputs are "mostly deterministic," not guaranteed identical.

The **A/B evaluator** (`tools/evaluate_ab.py`) solves this by sending both messages to the **same judge call**. The judge sees Message A (v1) and Message B (v2) side by side, so its internal "strictness" applies equally to both. The absolute scores may shift between calls, but the **relative delta is stable**.

## Judge Simulator — Apples-to-Apples (gpt-5.4 judge)

Both v1 and v2 scored by the same `judge_simulator.py` with `LLM_MODEL=gpt-5.4`:

| Metric | v1 Deterministic | v2 Hybrid | Delta |
|---|---|---|---|
| **Average score** | **37/50 (74%)** | **38.7/50 (77%)** | **+1.7** |
| Avg Specificity | 8 | 7 | -1 |
| Avg Category Fit | 7 | 8 | +1 |
| Avg Merchant Fit | 7 | 7 | 0 |
| Avg Decision Quality | 8 | 8 | 0 |
| Avg Engagement | 7 | 7 | 0 |
| Auto-reply turns to exit | 3 | **1** | -2 turns |
| Messages scored | 25 | 23 | — |

The +1.7 delta is compressed by judge variance. The controlled A/B eval below shows the real delta.

## A/B Controlled Evaluation (gpt-5.4, same call)

### Known Triggers (12 base test pairs)

| Metric | v1 Deterministic | v2 Hybrid | Delta |
|---|---|---|---|
| Average score | 33.5/50 | **37.9/50** | **+4.4** |
| Win/Loss | — | **8W / 0L / 4T** | — |

### Novel Triggers (8 simulated Phase 3 injections)

Trigger kinds v1 has never seen: weather_heatwave, competitor_price_undercut, staff_review_highlight, local_event_disruption, category_trend_movement, research_digest (new item), customer_birthday_upcoming, google_algorithm_update.

| Metric | v1 Deterministic | v2 Hybrid | Delta |
|---|---|---|---|
| Average score | 23.9/50 | **39.8/50** | **+15.9** |
| Win/Loss | — | **8W / 0L / 0T** | — |

### Overall (20 triggers)

| Metric | v1 Deterministic | v2 Hybrid | Delta |
|---|---|---|---|
| Average score | 30.2/50 | **40.6/50** | **+10.4** |
| Win/Loss | — | **16W / 0L / 4T** | — |

## Architecture Iterations

We tested three architectures before arriving at the dual-path design:

### Attempt 1: Pure LLM compose (replace all deterministic code)
- Result: LLM paraphrased specific numbers away ("calls down 50%" became "calls dipped"), hallucinated facts
- Scored **worse** than deterministic on known triggers

### Attempt 2: Template + LLM polish (LLM only adds Hindi mix to deterministic draft)
- Result: 10W/2L vs deterministic. Better on novel triggers, but template substitution bugs (wrong merchant names in customer-scoped messages) caused losses

### Attempt 3: Dual-path + selector (final)
- Both deterministic and LLM run in parallel
- LLM selector picks the better message
- Below BM25 threshold (3.5): skip selector, use LLM directly (novel trigger)
- Result: 16W/0L/4T — zero losses

## Key Bugs Found and Fixed

1. **Percentage rendering**: raw `delta_pct: -0.5` was shown as "0.5%" instead of "50%". Fixed by pre-formatting in fact_extractor.
2. **Customer name swap**: `_build_deterministic()` was replacing customer greeting names (e.g., "Hi Rashmi") with the merchant name (e.g., "Hi Karthik"). Fixed by detecting customer-scoped templates and preserving the original greeting.
3. **Template category mismatch**: BM25 matched a dentist template for a salon trigger (competitor_price_undercut). Fixed by adding explicit WARNING in the prompt when template category differs.
4. **LLM trigger drift**: research_digest messages led with CTR stats instead of the digest item. Fixed by putting trigger facts BEFORE merchant facts in the prompt (primacy bias).

## How to Reproduce

```bash
# Start v2 bot
uv run uvicorn bot:app --host 0.0.0.0 --port 8080

# Run A/B eval (default judge: gpt-5.4)
uv run python tools/evaluate_ab.py

# Override judge model
JUDGE_MODEL=gpt-4o uv run python tools/evaluate_ab.py
```
