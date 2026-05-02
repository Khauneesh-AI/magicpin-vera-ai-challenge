#!/usr/bin/env python3
"""A/B evaluation: compares deterministic v1 messages (from submission.jsonl)
against the live hybrid v2 bot, using a strong LLM judge for controlled scoring.

WHY THIS EXISTS:
The judge_simulator.py scores one bot at a time, so comparing two runs has
high variance (different LLM judge calls, different randomness). This script
sends BOTH messages (A and B) to the SAME judge call, eliminating judge
variance and producing a reliable delta.

It also tests against synthetic Phase 3 novel triggers that the deterministic
solution has no templates for, showing where the LLM path adds real value.

USAGE:
    # Start the v2 bot first:
    uv run uvicorn bot:app --host 0.0.0.0 --port 8080

    # Then run this script:
    OPENAI_API_KEY=sk-... uv run python tools/evaluate_ab.py

    # Override judge model (default: gpt-5.4):
    JUDGE_MODEL=gpt-4o uv run python tools/evaluate_ab.py

WHAT IT MEASURES:
    - Known triggers (12): deterministic v1 vs hybrid v2 on the base test set
    - Novel triggers (8): deterministic fallback vs hybrid v2 on Phase 3 injections
    - Per-dimension breakdown: specificity, category_fit, merchant_fit, trigger_relevance, engagement
    - Win/loss record and average delta
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Ensure vera_bot is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import AsyncOpenAI

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4")
ROOT = Path(__file__).resolve().parents[1]

EVAL_PROMPT = """\
You are a STRICT judge for the magicpin AI Challenge. Score BOTH messages for the SAME trigger.
5=mediocre, 7=good, 9+=excellent. Be harsh.

Category: {category} (voice: {voice}) | Merchant: {merchant_name} ({locality}, {city})
Languages: {languages} | Trigger: {trigger_kind} | Offers: {offers}

A (deterministic v1): "{msg_a}"

B (hybrid v2 — dual-path BM25+LLM): "{msg_b}"

Score each on 5 dimensions (0-10): specificity, category_fit, merchant_fit, trigger_relevance, engagement.
Return ONLY valid JSON:
{{"a_total":X,"b_total":X,"key_difference":"one sentence"}}
where total = specificity + category_fit + merchant_fit + trigger_relevance + engagement (max 50)"""


def deterministic_fallback(merchant: dict, trigger: dict) -> str:
    """Simulates what v1's generic fallback produces for unknown triggers."""
    ident = merchant.get("identity", {})
    name = ident.get("owner_first_name", "Merchant")
    if merchant.get("category_slug") == "dentists" and not name.lower().startswith("dr"):
        name = f"Dr. {name}"
    mn = ident.get("name", "")
    loc = ident.get("locality", "")
    city = ident.get("city", "")
    perf = merchant.get("performance", {})
    kind = trigger.get("kind", "").replace("_", " ")
    payload = trigger.get("payload", {})
    facts = []
    for k, v in list(payload.items())[:2]:
        if k not in ("merchant_id", "customer_id", "category") and v:
            facts.append(f'{k.replace("_", " ")} {v}')
    body = f"{name}, quick signal for {mn}"
    if loc and city:
        body += f' in {loc}, {city}; {perf.get("views", "?")} views/{perf.get("calls", "?")} calls'
    body += f": {kind}"
    if facts:
        body += f' ({"; ".join(facts)})'
    body += ". Want me to turn this into one customer-facing post or WhatsApp draft?"
    return body


async def main() -> None:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load data
    cats = {}
    for name in ["dentists", "restaurants", "gyms", "salons", "pharmacies"]:
        cats[name] = json.loads((ROOT / f"expanded/categories/{name}.json").read_text())
    seed_m = json.loads((ROOT / "dataset/merchants_seed.json").read_text())
    mers = {m["merchant_id"]: m for m in seed_m.get("merchants", [])}
    seed_t = json.loads((ROOT / "dataset/triggers_seed.json").read_text())
    trgs = {t["id"]: t for t in seed_t.get("triggers", [])}
    old_sub = {}
    for line in (ROOT / "submission.jsonl").open():
        row = json.loads(line)
        old_sub[row["test_id"]] = row
    pairs = json.loads((ROOT / "expanded/test_pairs.json").read_text())["pairs"]

    # Phase 3 data
    novel_path = ROOT / "expanded/phase3_triggers.json"
    novel = json.loads(novel_path.read_text()) if novel_path.exists() else []
    cust_path = ROOT / "expanded/phase3_customer.json"
    phase3_cust = json.loads(cust_path.read_text()) if cust_path.exists() else None
    digest_path = ROOT / "expanded/phase3_new_digest.json"
    if digest_path.exists():
        cats["dentists"]["digest"].append(json.loads(digest_path.read_text()))

    from vera_bot.composer import compose

    async def judge(msg_a: str, msg_b: str, mer: dict, trg: dict, cat: dict) -> dict | None:
        prompt = EVAL_PROMPT.format(
            category=cat["slug"],
            voice=cat.get("voice", {}).get("tone", ""),
            merchant_name=mer.get("identity", {}).get("name", ""),
            locality=mer.get("identity", {}).get("locality", ""),
            city=mer.get("identity", {}).get("city", ""),
            languages=mer.get("identity", {}).get("languages", []),
            trigger_kind=trg["kind"],
            offers=[o["title"] for o in mer.get("offers", []) if o.get("status") == "active"],
            msg_a=msg_a[:300],
            msg_b=msg_b[:300],
        )
        resp = await client.responses.create(
            model=JUDGE_MODEL,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        match = re.search(r"\{[\s\S]*?\}", resp.output_text)
        return json.loads(match.group()) if match else None

    a_known, b_known, a_novel, b_novel = [], [], [], []

    # --- Known triggers ---
    print("=== KNOWN TRIGGERS (v1 submission.jsonl vs v2 live) ===")
    known_pairs = [p for p in pairs if p["trigger_id"] in trgs and p["merchant_id"] in mers][:12]
    for pair in known_pairs:
        tid, mid, test_id = pair["trigger_id"], pair["merchant_id"], pair["test_id"]
        trg, mer = trgs[tid], mers[mid]
        cat = cats[mer["category_slug"]]
        new_msg = await compose(cat, mer, trg, None)
        old_msg = old_sub.get(test_id, {}).get("body", "")
        scores = await judge(old_msg, new_msg["body"], mer, trg, cat)
        if scores:
            a, b = scores["a_total"], scores["b_total"]
            a_known.append(a)
            b_known.append(b)
            d = b - a
            mk = ">>>" if d >= 3 else "<<<" if d <= -3 else "   "
            print(f'{mk} {test_id} {trg["kind"][:22]:22} A={a:2} B={b:2} d={d:+d}')

    # --- Novel triggers ---
    if novel:
        print()
        print("=== NOVEL TRIGGERS (v1 generic fallback vs v2 LLM compose) ===")
        for trg in novel:
            mid = trg["merchant_id"]
            if mid not in mers:
                continue
            mer = mers[mid]
            cat = cats[mer["category_slug"]]
            cust = phase3_cust if trg.get("customer_id") == "c_phase3_neha" else None
            msg_a = deterministic_fallback(mer, trg)
            msg_b_r = await compose(cat, mer, trg, cust)
            scores = await judge(msg_a, msg_b_r["body"], mer, trg, cat)
            if scores:
                a, b = scores["a_total"], scores["b_total"]
                a_novel.append(a)
                b_novel.append(b)
                d = b - a
                mk = ">>>" if d >= 3 else "<<<" if d <= -3 else "   "
                print(f'{mk} {trg["kind"][:28]:28} A={a:2} B={b:2} d={d:+d}')

    # --- Summary ---
    print()
    print("=" * 60)
    if a_known:
        ak = sum(a_known) / len(a_known)
        bk = sum(b_known) / len(b_known)
        kw = sum(1 for a, b in zip(a_known, b_known) if b > a)
        kl = sum(1 for a, b in zip(a_known, b_known) if b < a)
        print(f"KNOWN ({len(a_known):2}):  A={ak:.1f}  B={bk:.1f}  delta={bk - ak:+.1f}  record={kw}W/{kl}L")
    if a_novel:
        an = sum(a_novel) / len(a_novel)
        bn = sum(b_novel) / len(b_novel)
        nw = sum(1 for a, b in zip(a_novel, b_novel) if b > a)
        nl = sum(1 for a, b in zip(a_novel, b_novel) if b < a)
        print(f"NOVEL ({len(a_novel):2}):  A={an:.1f}  B={bn:.1f}  delta={bn - an:+.1f}  record={nw}W/{nl}L")
    all_a = a_known + a_novel
    all_b = b_known + b_novel
    if all_a:
        ta = sum(all_a) / len(all_a)
        tb = sum(all_b) / len(all_b)
        tw = sum(1 for a, b in zip(all_a, all_b) if b > a)
        tl = sum(1 for a, b in zip(all_a, all_b) if b < a)
        tt = len(all_a) - tw - tl
        print(f"OVERALL({len(all_a):2}):  A={ta:.1f}  B={tb:.1f}  delta={tb - ta:+.1f}  record={tw}W/{tl}L/{tt}T")
    print()
    print(f"Judge model: {JUDGE_MODEL}")


if __name__ == "__main__":
    asyncio.run(main())
