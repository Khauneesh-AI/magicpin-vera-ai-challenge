# Vera Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Magicpin Vera challenge bot into a clean deterministic Phase 1 submission with validation-first safeguards.

**Architecture:** Keep `compose(category, merchant, trigger, customer=None)` pure and deterministic. Put API state in `vera_bot.store`, HTTP routing in `vera_bot.routes`, deterministic message logic in `vera_bot.composer` and `vera_bot.trigger_handlers`, and cleanup in `vera_bot.validators`.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn, standard-library `unittest`, standard-library `json`/`pathlib`.

---

## File Map

Create:

- `vera_bot/__init__.py`: package marker and public version.
- `vera_bot/config.py`: metadata, limits, URL policy, allowed `send_as` values.
- `vera_bot/schemas.py`: Pydantic models for context, tick, reply, action, health, metadata.
- `vera_bot/store.py`: in-memory state and helpers for versioned context, suppression, recent body hashes, conversations.
- `vera_bot/validators.py`: deterministic output cleanup and contract validation.
- `vera_bot/trigger_handlers.py`: deterministic merchant/customer trigger-family handlers.
- `vera_bot/composer.py`: pure `compose()` dispatching to handlers and validators.
- `vera_bot/reply_handler.py`: deterministic reply state machine.
- `vera_bot/routes.py`: all `/v1/*` endpoints.
- `vera_bot/main.py`: FastAPI app factory and `app`.
- `tools/generate_submission.py`: generate `submission.jsonl` from `expanded/test_pairs.json`.
- `tools/validate_submission.py`: validate JSONL, deterministic replay, no URLs, no `None`.
- `tools/smoke_endpoints.py`: smoke-test a running local/remote API.
- `tests/test_contract.py`: fast unit tests for composition, validation, context versioning, tick suppression, reply decisions.

Modify:

- `bot.py`: replace current prototype with public shim after package passes tests.
- `README.md`: update final one-page approach and run instructions.
- `requirements.txt`: keep only `fastapi`, `uvicorn[standard]`, `pydantic`.
- `submission.jsonl`: regenerate after clean package is wired.

Reference only:

- Current `bot.py`: source for proven message templates and handler behavior.
- `challenge-brief.md`: composition contract and scoring rubric.
- `challenge-testing-brief.md`: HTTP contract and judge harness behavior.
- `expanded/test_pairs.json`: canonical 30 test rows.

---

## Task 1: Validation Harness Against Current Prototype

**Files:**
- Create: `tools/generate_submission.py`
- Create: `tools/validate_submission.py`
- Create: `tests/test_contract.py`

- [ ] **Step 1: Create `tools/generate_submission.py`**

Use this script structure:

```python
from __future__ import annotations

import json
from pathlib import Path

from bot import compose

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "expanded"
OUT = ROOT / "submission.jsonl"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def context_path(scope: str, context_id: str) -> Path:
    folders = {
        "category": "categories",
        "merchant": "merchants",
        "customer": "customers",
        "trigger": "triggers",
    }
    return EXPANDED / folders[scope] / f"{context_id}.json"


def build_rows() -> list[dict]:
    pairs = load_json(EXPANDED / "test_pairs.json")
    rows: list[dict] = []
    for pair in pairs:
        category = load_json(context_path("category", pair["category_id"]))
        merchant = load_json(context_path("merchant", pair["merchant_id"]))
        trigger = load_json(context_path("trigger", pair["trigger_id"]))
        customer = None
        if pair.get("customer_id"):
            customer = load_json(context_path("customer", pair["customer_id"]))
        msg = compose(category, merchant, trigger, customer)
        rows.append({"test_id": pair["test_id"], **msg})
    return rows


def main() -> None:
    rows = build_rows()
    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `tools/validate_submission.py`**

Validate exact required keys, 30 rows, deterministic replay, no URLs, no `None`, accepted `send_as`, and max body length:

```python
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from tools.generate_submission import build_rows

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission.jsonl"
REQUIRED = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
URL_RE = re.compile(r"https?://|www\.", re.I)
MAX_BODY_CHARS = 320


def fail(message: str) -> None:
    raise AssertionError(message)


def load_rows() -> list[dict]:
    if not SUBMISSION.exists():
        fail("submission.jsonl does not exist; run tools/generate_submission.py")
    return [json.loads(line) for line in SUBMISSION.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rows(rows: list[dict]) -> None:
    if len(rows) != 30:
        fail(f"expected 30 rows, got {len(rows)}")
    seen_ids = set()
    for idx, row in enumerate(rows, start=1):
        missing = REQUIRED - set(row)
        if missing:
            fail(f"row {idx} missing keys: {sorted(missing)}")
        if row["test_id"] in seen_ids:
            fail(f"duplicate test_id {row['test_id']}")
        seen_ids.add(row["test_id"])
        body = str(row["body"])
        if len(body) > MAX_BODY_CHARS:
            fail(f"{row['test_id']} body too long: {len(body)}")
        if "None" in body:
            fail(f"{row['test_id']} body leaks None")
        if URL_RE.search(body) or URL_RE.search(str(row["cta"])):
            fail(f"{row['test_id']} contains URL")
        if row["send_as"] not in {"vera", "merchant_on_behalf"}:
            fail(f"{row['test_id']} invalid send_as {row['send_as']!r}")


def main() -> None:
    rows = load_rows()
    validate_rows(rows)
    replay = build_rows()
    validate_rows(replay)
    if rows != replay:
        fail("deterministic replay mismatch; regenerate submission or fix compose()")
    print("validation passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
```

- [ ] **Step 3: Create initial `tests/test_contract.py`**

Use standard-library `unittest` so no new dependency is needed:

```python
from __future__ import annotations

import unittest

from tools.generate_submission import build_rows
from tools.validate_submission import REQUIRED, URL_RE, validate_rows


class SubmissionContractTests(unittest.TestCase):
    def test_generated_rows_match_contract(self) -> None:
        rows = build_rows()
        validate_rows(rows)

    def test_deterministic_generation(self) -> None:
        self.assertEqual(build_rows(), build_rows())

    def test_required_keys_are_expected(self) -> None:
        self.assertEqual(
            REQUIRED,
            {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"},
        )

    def test_url_regex_catches_urls(self) -> None:
        self.assertIsNotNone(URL_RE.search("https://example.com"))
        self.assertIsNotNone(URL_RE.search("www.example.com"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run validation against current prototype**

Run:

```powershell
python tools\generate_submission.py
python tools\validate_submission.py
python -m unittest tests.test_contract -v
```

Expected: all commands pass. If the current prototype fails no-URL validation, record the failure and continue; the clean rebuild must pass it.

---

## Task 2: Package Skeleton And Shared Types

**Files:**
- Create: `vera_bot/__init__.py`
- Create: `vera_bot/config.py`
- Create: `vera_bot/schemas.py`
- Create: `vera_bot/store.py`
- Modify: `tests/test_contract.py`

- [ ] **Step 1: Add package files**

`vera_bot/__init__.py`:

```python
__version__ = "1.0.0"
```

`vera_bot/config.py`:

```python
TEAM_NAME = "Vera Phase 1"
TEAM_MEMBERS: list[str] = []
CONTACT_EMAIL = "replace-before-submit@example.com"
APPROACH = "deterministic-python-rules"
MODEL_NAME = "deterministic-no-llm"
MAX_BODY_CHARS = 320
MAX_ACTIONS_PER_TICK = 20
ANTI_REPEAT_WINDOW = 5
URL_POLICY = "strip_all"
ALLOWED_SEND_AS = {"vera", "merchant_on_behalf"}
```

- [ ] **Step 2: Add Pydantic schemas**

`vera_bot/schemas.py` should define:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = 1
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickBody(BaseModel):
    now: str | None = None
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: Literal["merchant", "customer", "system"] = "merchant"
    message: str
    received_at: str | None = None
    turn_number: int = 1


class ComposedMessage(BaseModel):
    body: str
    cta: str
    send_as: Literal["vera", "merchant_on_behalf"] = "vera"
    suppression_key: str
    rationale: str


class Action(ComposedMessage):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    trigger_id: str


class ReplyDecision(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str | None = None
    cta: str | None = None
    rationale: str
```

- [ ] **Step 3: Add in-memory store**

`vera_bot/store.py` should include `Store.push_context()`, `Store.get_payload()`, `Store.mark_sent()`, `Store.was_sent()`, `Store.remember_body()`, `Store.seen_recent_body()`, and `Store.add_turn()`. Suppression keys must be stored as `(merchant_id, suppression_key)`.

- [ ] **Step 4: Add store tests**

Extend `tests/test_contract.py` with:

```python
from vera_bot.store import Store


class StoreTests(unittest.TestCase):
    def test_context_stale_version_does_not_overwrite(self) -> None:
        store = Store()
        first = store.push_context("merchant", "m1", 2, {"name": "new"})
        stale = store.push_context("merchant", "m1", 1, {"name": "old"})
        self.assertTrue(first["accepted"])
        self.assertFalse(stale["accepted"])
        self.assertEqual(stale["reason"], "stale_version")
        self.assertEqual(store.get_payload("merchant", "m1"), {"name": "new"})

    def test_suppression_is_merchant_scoped(self) -> None:
        store = Store()
        store.mark_sent("m1", "same-key")
        self.assertTrue(store.was_sent("m1", "same-key"))
        self.assertFalse(store.was_sent("m2", "same-key"))
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m unittest tests.test_contract -v
```

Expected: store tests pass after `store.py` is implemented.

---

## Task 3: Validators And Pure Composer

**Files:**
- Create: `vera_bot/validators.py`
- Create: `vera_bot/trigger_handlers.py`
- Create: `vera_bot/composer.py`
- Modify: `tests/test_contract.py`

- [ ] **Step 1: Write validator tests**

Add tests for URL stripping, `None` cleanup, length cap, and `send_as` normalization:

```python
from vera_bot.validators import finalize_message


class ValidatorTests(unittest.TestCase):
    def test_finalize_strips_urls_and_none(self) -> None:
        msg = {
            "body": "See https://example.com None",
            "cta": "Open www.example.com",
            "send_as": "bad",
            "suppression_key": "",
            "rationale": "",
        }
        out = finalize_message(msg, fallback_key="fallback:key")
        self.assertNotRegex(out["body"], r"https?://|www\.")
        self.assertNotIn("None", out["body"])
        self.assertEqual(out["send_as"], "vera")
        self.assertEqual(out["suppression_key"], "fallback:key")
```

- [ ] **Step 2: Implement `finalize_message()`**

The function signature:

```python
def finalize_message(message: dict, *, fallback_key: str) -> dict[str, str]:
    ...
```

It returns exactly `body`, `cta`, `send_as`, `suppression_key`, `rationale`.

- [ ] **Step 3: Port trigger logic from prototype**

Move the current deterministic message behavior from `bot.py` into `vera_bot/trigger_handlers.py`. Keep helper functions small:

- `merchant_message(category, merchant, trigger) -> dict`
- `customer_message(category, merchant, trigger, customer) -> dict`
- `missing_customer_message(merchant, trigger) -> dict`
- helper formatters such as `pct()`, `first_name()`, `merchant_short_name()`, `best_offer()`, `find_digest_item()`

- [ ] **Step 4: Implement pure `compose()`**

`vera_bot/composer.py`:

```python
from __future__ import annotations

from typing import Any

from vera_bot.trigger_handlers import customer_message, merchant_message, missing_customer_message
from vera_bot.validators import finalize_message


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, str]:
    if customer is None and trigger.get("scope") == "customer":
        raw = missing_customer_message(merchant, trigger)
    elif customer is not None or trigger.get("scope") == "customer":
        raw = customer_message(category, merchant, trigger, customer or {})
    else:
        raw = merchant_message(category, merchant, trigger)
    fallback_key = str(trigger.get("suppression_key") or f"{trigger.get('kind', 'unknown')}:{trigger.get('id', '')}")
    return finalize_message(raw, fallback_key=fallback_key)
```

- [ ] **Step 5: Point tests at new composer**

Update `tools/generate_submission.py` to import `compose` from `vera_bot.composer` once the new composer passes direct checks.

- [ ] **Step 6: Run validation**

Run:

```powershell
python tools\generate_submission.py
python tools\validate_submission.py
python -m unittest tests.test_contract -v
```

Expected: validation passes with no URLs and deterministic replay.

---

## Task 4: API Routes And Public Shim

**Files:**
- Create: `vera_bot/routes.py`
- Create: `vera_bot/main.py`
- Modify: `bot.py`
- Modify: `tests/test_contract.py`

- [ ] **Step 1: Add route-level tests using FastAPI TestClient**

Add tests for health, metadata, stale context, and tick suppression:

```python
from fastapi.testclient import TestClient
from vera_bot.main import app


class ApiTests(unittest.TestCase):
    def test_health_and_metadata(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/v1/healthz").status_code, 200)
        metadata = client.get("/v1/metadata").json()
        self.assertEqual(metadata["model"], "deterministic-no-llm")

    def test_context_stale_response_shape(self) -> None:
        client = TestClient(app)
        body = {"scope": "merchant", "context_id": "m-test", "version": 2, "payload": {"merchant_id": "m-test"}}
        self.assertTrue(client.post("/v1/context", json=body).json()["accepted"])
        stale = {**body, "version": 1, "payload": {"merchant_id": "old"}}
        response = client.post("/v1/context", json=stale)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reason"], "stale_version")
```

- [ ] **Step 2: Implement `vera_bot/routes.py`**

Define one router with all five endpoints:

- `GET /healthz`
- `GET /metadata`
- `POST /context`
- `POST /tick`
- `POST /reply`

Use a module-level `store = Store()` for Phase 1.

- [ ] **Step 3: Implement tick orchestration**

`/v1/tick` must iterate over every `available_triggers` item and break only after collecting `MAX_ACTIONS_PER_TICK` valid actions. It must skip missing contexts and duplicate `(merchant_id, suppression_key)` sends.

- [ ] **Step 4: Implement `vera_bot/main.py`**

```python
from __future__ import annotations

from fastapi import FastAPI

from vera_bot.routes import router

app = FastAPI(title="Magicpin Vera Phase 1 Bot", version="1.0.0")
app.include_router(router, prefix="/v1")
```

- [ ] **Step 5: Replace root `bot.py` with shim**

```python
from vera_bot.composer import compose
from vera_bot.main import app

__all__ = ["app", "compose"]
```

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m unittest tests.test_contract -v
python tools\validate_submission.py
```

Expected: all tests and validation pass.

---

## Task 5: Reply Handler

**Files:**
- Create: `vera_bot/reply_handler.py`
- Modify: `vera_bot/routes.py`
- Modify: `tests/test_contract.py`

- [ ] **Step 1: Add reply tests**

Add tests for auto-reply, hostile/opt-out, off-topic, explicit commit, and ambiguous reply:

```python
from vera_bot.reply_handler import decide_reply


class ReplyHandlerTests(unittest.TestCase):
    def test_auto_reply_waits_or_ends(self) -> None:
        decision = decide_reply(
            conversation_id="c1",
            message="Aapki jaankari ke liye bahut-bahut shukriya",
            history=[],
        )
        self.assertIn(decision["action"], {"send", "wait", "end"})
        self.assertIn("auto", decision["rationale"].lower())

    def test_explicit_commit_sends_action_confirmation(self) -> None:
        decision = decide_reply(conversation_id="c2", message="ok go ahead", history=[])
        self.assertEqual(decision["action"], "send")
        self.assertIn("confirm", decision["rationale"].lower())
```

- [ ] **Step 2: Implement detection helpers**

Implement:

- `is_hostile_or_opt_out(text) -> bool`
- `is_off_topic(text) -> bool`
- `is_auto_reply(text) -> bool`
- `is_commit(text) -> bool`
- `decide_reply(conversation_id, message, history) -> dict`

- [ ] **Step 3: Wire `/v1/reply`**

Route should append inbound turn, call `decide_reply()`, append outbound turn when action is `send`, and return the decision.

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m unittest tests.test_contract -v
```

Expected: reply handler tests pass.

---

## Task 6: Smoke Tool And End-To-End API Check

**Files:**
- Create: `tools/smoke_endpoints.py`
- Modify: `tests/test_contract.py` if route behavior exposes a regression.

- [ ] **Step 1: Create smoke script**

Use standard-library `urllib.request` so no HTTP dependency is added:

```python
from __future__ import annotations

import json
import sys
import urllib.request


def request(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    health = request("GET", f"{base}/v1/healthz")
    metadata = request("GET", f"{base}/v1/metadata")
    context = request("POST", f"{base}/v1/context", {
        "scope": "category",
        "context_id": "smoke",
        "version": 1,
        "payload": {"slug": "smoke", "display_name": "Smoke"},
    })
    tick = request("POST", f"{base}/v1/tick", {"available_triggers": []})
    reply = request("POST", f"{base}/v1/reply", {
        "conversation_id": "smoke-convo",
        "from_role": "merchant",
        "message": "ok",
    })
    assert health["status"] == "ok"
    assert "model" in metadata
    assert context["accepted"] is True
    assert tick["actions"] == []
    assert reply["action"] in {"send", "wait", "end"}
    print("smoke passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run local API**

Run in one terminal:

```powershell
uvicorn bot:app --host 127.0.0.1 --port 8000
```

Run in another terminal:

```powershell
python tools\smoke_endpoints.py http://127.0.0.1:8000
```

Expected: `smoke passed`.

---

## Task 7: Final Artifacts

**Files:**
- Modify: `README.md`
- Modify: `submission.jsonl`
- Modify: `requirements.txt`

- [ ] **Step 1: Regenerate submission**

Run:

```powershell
python tools\generate_submission.py
python tools\validate_submission.py
```

Expected: `submission.jsonl` has 30 valid rows and validation passes.

- [ ] **Step 2: Update README**

Use this structure:

````markdown
# Vera Challenge Submission

## Approach
Deterministic Python composer for Magicpin Vera merchant engagement. The bot uses category, merchant, trigger, and optional customer context to produce specific WhatsApp-style messages without live LLM calls.

## Tradeoffs
- Deterministic rules are stable and fast, but less flexible than a live LLM.
- Missing facts produce conservative fallbacks rather than invented details.
- URLs are stripped in Phase 1 to avoid accidental fabrication.

## Run Locally
```powershell
python tools\generate_submission.py
python tools\validate_submission.py
uvicorn bot:app --host 0.0.0.0 --port 8000
python tools\smoke_endpoints.py http://127.0.0.1:8000
```
````

- [ ] **Step 3: Final verification**

Run:

```powershell
python -m unittest tests.test_contract -v
python tools\generate_submission.py
python tools\validate_submission.py
python -m py_compile bot.py vera_bot\*.py tools\*.py tests\*.py
```

Expected: all commands pass.

---

## Task 8: Optional Git Setup

**Files:**
- Repository metadata only.

- [ ] **Step 1: Initialize git if the user wants commits**

Run only with user approval:

```powershell
git init
git add .
git commit -m "docs: add vera phase 1 design and implementation plan"
```

Expected: repository exists and the planning docs are committed.

This workspace is currently not a git repository, so implementation can proceed without commits unless the user wants commit history.

---

## Self-Review

Spec coverage:

- Pure deterministic `compose()`: Task 3.
- Flat `vera_bot/` package: Task 2 through Task 5.
- Required endpoints: Task 4.
- Submission generation and validation: Task 1 and Task 7.
- In-memory store and versioned context: Task 2 and Task 4.
- Suppression and anti-repeat foundation: Task 2 and Task 4.
- Reply handling: Task 5.
- Smoke testing: Task 6.
- No live LLM: enforced by file map and config.
- No URLs: Task 1 and Task 3 validators.

Placeholder scan:

- No incomplete sections remain.
- All created files have exact paths.
- Every task has verification commands.

Type consistency:

- `compose()` returns `dict[str, str]` throughout.
- `ContextBody`, `TickBody`, and `ReplyBody` are defined before route usage.
- `Store` methods named in tests match the planned store interface.
