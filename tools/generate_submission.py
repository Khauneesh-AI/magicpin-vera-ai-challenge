from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "expanded"
OUT = ROOT / "submission.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera_bot.composer import compose


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
    pairs_data = load_json(EXPANDED / "test_pairs.json")
    pairs = pairs_data["pairs"] if isinstance(pairs_data, dict) else pairs_data
    rows: list[dict] = []
    for pair in pairs:
        merchant = load_json(context_path("merchant", pair["merchant_id"]))
        category_id = pair.get("category_id") or merchant["category_slug"]
        category = load_json(context_path("category", category_id))
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
