from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission.jsonl"
REQUIRED = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
URL_RE = re.compile(r"https?://|www\.", re.I)
MAX_BODY_CHARS = 320

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_submission import build_rows


def fail(message: str) -> None:
    raise AssertionError(message)


def load_rows() -> list[dict]:
    if not SUBMISSION.exists():
        fail("submission.jsonl does not exist; run tools/generate_submission.py")
    return [
        json.loads(line)
        for line in SUBMISSION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
