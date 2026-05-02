"""Contract and unit tests for the Vera bot v2 (hybrid BM25 + LLM).

Tests that require LLM calls mock the OpenAI client. Tests for
deterministic components (fact extraction, BM25, validation, store)
run without mocks.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from vera_bot.fact_extractor import (
    active_offer,
    city_locality,
    customer_name,
    extract_all_facts,
    find_digest_item,
    first_name,
    get_peer_comparison,
)
from vera_bot.llm_client import ComposedMessage, ReplyClassification
from vera_bot.main import app
from vera_bot.prompts import build_compose_system, build_compose_user
from vera_bot.reply_handler import (
    is_auto_reply_keyword,
    is_exact_opt_out,
    repeated_in_history,
    _keyword_fallback,
)
from vera_bot.retrieval import TemplateIndex, index as global_index
from vera_bot.schemas import ContextBody
from vera_bot.store import Store
from vera_bot.validators import finalize_message
from vera_bot import routes

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "expanded"


def load_context(scope: str, context_id: str) -> dict:
    folder = {"category": "categories", "merchant": "merchants", "trigger": "triggers", "customer": "customers"}[scope]
    return json.loads((EXPANDED / folder / f"{context_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fact Extractor Tests
# ---------------------------------------------------------------------------


class FactExtractorTests(unittest.TestCase):
    def test_first_name_adds_dr_prefix_for_dentists(self) -> None:
        merchant = {"identity": {"owner_first_name": "Meera"}, "category_slug": "dentists"}
        self.assertEqual(first_name(merchant), "Dr. Meera")

    def test_first_name_no_prefix_for_restaurants(self) -> None:
        merchant = {"identity": {"owner_first_name": "Suresh"}, "category_slug": "restaurants"}
        self.assertEqual(first_name(merchant), "Suresh")

    def test_city_locality_both(self) -> None:
        merchant = {"identity": {"locality": "Lajpat Nagar", "city": "Delhi"}}
        self.assertEqual(city_locality(merchant), "Lajpat Nagar, Delhi")

    def test_active_offer_picks_active(self) -> None:
        merchant = {
            "offers": [
                {"title": "Expired Offer", "status": "expired"},
                {"title": "Dental Cleaning @ ₹299", "status": "active"},
            ]
        }
        self.assertEqual(active_offer(merchant), "Dental Cleaning @ ₹299")

    def test_find_digest_item_by_id(self) -> None:
        category = {"digest": [{"id": "d1", "title": "Item 1"}, {"id": "d2", "title": "Item 2"}]}
        trigger = {"kind": "research_digest", "payload": {"top_item_id": "d2"}}
        self.assertEqual(find_digest_item(category, trigger)["title"], "Item 2")

    def test_customer_name_strips_parenthetical(self) -> None:
        customer = {"identity": {"name": "Karthik (parent: Sumitra)"}}
        self.assertEqual(customer_name(customer), "Karthik")

    def test_peer_comparison_below(self) -> None:
        merchant = {"performance": {"ctr": 0.021}}
        category = {"peer_stats": {"avg_ctr": 0.030}}
        result = get_peer_comparison(merchant, category)
        self.assertEqual(result["ctr_vs_peer"], "below")

    def test_extract_all_facts_has_required_keys(self) -> None:
        cat = load_context("category", "dentists")
        mer = load_context("merchant", "m_001_drmeera_dentist_delhi")
        trg = load_context("trigger", "trg_001_research_digest_dentists")
        facts = extract_all_facts(cat, mer, trg)
        self.assertIn("merchant", facts)
        self.assertIn("trigger", facts)
        self.assertIn("customer", facts)
        self.assertIn("category_slug", facts)
        self.assertIn("category_voice", facts)
        self.assertEqual(facts["merchant"]["name"], "Dr. Meera")


# ---------------------------------------------------------------------------
# BM25 Retrieval Tests
# ---------------------------------------------------------------------------


class RetrievalTests(unittest.TestCase):
    def test_global_index_loaded(self) -> None:
        self.assertGreater(len(global_index.entries), 20)

    def test_query_returns_relevant_matches(self) -> None:
        trigger = {"kind": "perf_dip", "payload": {"metric": "calls", "delta_pct": -0.5}}
        category = {"slug": "dentists"}
        matches = global_index.query(trigger, category, top_n=2)
        self.assertEqual(len(matches), 2)
        entry, score = matches[0]
        self.assertIn("perf_dip", entry["trigger_kind"])
        self.assertGreater(score, 0)

    def test_customer_trigger_retrieves_customer_scope(self) -> None:
        trigger = {"kind": "recall_due", "scope": "customer", "payload": {"service_due": "cleaning"}}
        category = {"slug": "dentists"}
        matches = global_index.query(trigger, category, top_n=2)
        self.assertTrue(any(entry["scope"] == "customer" for entry, _ in matches))


# ---------------------------------------------------------------------------
# Prompts Tests
# ---------------------------------------------------------------------------


class PromptsTests(unittest.TestCase):
    def test_compose_system_includes_voice(self) -> None:
        category = {
            "slug": "dentists",
            "voice": {
                "tone": "peer_clinical",
                "vocab_allowed": ["fluoride", "caries"],
                "vocab_taboo": ["guaranteed"],
            },
        }
        prompt = build_compose_system(category)
        self.assertIn("peer_clinical", prompt)
        self.assertIn("fluoride", prompt)
        self.assertIn("guaranteed", prompt)

    def test_compose_user_polish_mode_includes_draft(self) -> None:
        facts = {
            "merchant": {"name": "Dr. Test", "location": "Delhi", "languages": ["en", "hi"]},
            "trigger": {"kind": "perf_dip", "urgency": 3, "scope": "merchant",
                        "suppression_key": "test:key", "payload": {}},
            "customer": None,
        }
        trigger = {"kind": "perf_dip", "payload": {"metric": "calls"}}
        templates = [({"trigger_kind": "perf_dip", "category": "dentists",
                       "sample_message": "Test message", "cta_type": "binary_yes_no",
                       "send_as": "vera", "compulsion_levers": ["specificity"]}, 10.0)]
        prompt = build_compose_user(facts, trigger, templates, mode="polish", draft_body="Draft body here")
        self.assertIn("Draft body here", prompt)
        self.assertIn("DRAFT MESSAGE", prompt)

    def test_compose_user_compose_mode_includes_facts(self) -> None:
        facts = {
            "merchant": {"name": "Dr. Test", "location": "Delhi"},
            "trigger": {"kind": "novel_trigger", "urgency": 2, "scope": "merchant", "payload": {}},
            "customer": None,
        }
        trigger = {"kind": "novel_trigger", "payload": {"metric": "calls"}}
        templates = [({"trigger_kind": "perf_dip", "sample_message": "Ref msg",
                       "cta_type": "open_ended", "send_as": "vera"}, 2.0)]
        prompt = build_compose_user(facts, trigger, templates, mode="compose")
        self.assertIn("Dr. Test", prompt)
        self.assertIn("novel_trigger", prompt)
        self.assertIn("BUSINESS NAME", prompt)  # few-shot guidance present


# ---------------------------------------------------------------------------
# Store Tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validator Tests
# ---------------------------------------------------------------------------


class StoreNewFeaturesTests(unittest.TestCase):
    def test_unanswered_nudge_counter(self) -> None:
        store = Store()
        store.record_bot_send("m1")
        store.record_bot_send("m1")
        self.assertEqual(store.get_unanswered_count("m1"), 2)
        self.assertFalse(store.should_stop_nudging("m1"))
        store.record_bot_send("m1")
        self.assertTrue(store.should_stop_nudging("m1"))
        store.record_merchant_reply("m1")
        self.assertEqual(store.get_unanswered_count("m1"), 0)
        self.assertFalse(store.should_stop_nudging("m1"))

    def test_language_detection_english(self) -> None:
        from vera_bot.store import detect_language
        self.assertEqual(detect_language("Yes, go ahead with the post"), "en")

    def test_language_detection_hindi(self) -> None:
        from vera_bot.store import detect_language
        self.assertEqual(detect_language("Haan chalega kar do"), "hi-en")

    def test_language_detection_devanagari(self) -> None:
        from vera_bot.store import detect_language
        self.assertEqual(detect_language("धन्यवाद, आगे बढ़ो"), "hi")

    def test_conversation_language_sticky(self) -> None:
        store = Store()
        store.detect_and_store_language("c1", "Yes do it")
        self.assertEqual(store.get_conversation_language("c1"), "en")
        store.detect_and_store_language("c1", "haan kar do bhai")
        self.assertEqual(store.get_conversation_language("c1"), "hi-en")
        # Once Hindi detected, stays hi-en even if next message is English
        store.detect_and_store_language("c1", "Ok proceed")
        self.assertEqual(store.get_conversation_language("c1"), "hi-en")

    def test_cadence_tracking(self) -> None:
        store = Store()
        store.record_send_time("m1")
        store.record_send_time("m1")
        self.assertEqual(store.get_sends_in_window("m1", 24), 2)

    def test_prompt_family_routing(self) -> None:
        from vera_bot.prompts import get_trigger_family
        self.assertEqual(get_trigger_family("research_digest"), "knowledge")
        self.assertEqual(get_trigger_family("perf_dip"), "performance")
        self.assertEqual(get_trigger_family("recall_due"), "customer")
        self.assertEqual(get_trigger_family("festival_upcoming"), "event")
        self.assertEqual(get_trigger_family("milestone_reached"), "social")
        self.assertEqual(get_trigger_family("dormant_with_vera"), "account")
        self.assertEqual(get_trigger_family("totally_new_trigger"), "fallback")


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


# ---------------------------------------------------------------------------
# Reply Handler Tests (Tier 1 only — no LLM)
# ---------------------------------------------------------------------------


class ReplyHandlerTier1Tests(unittest.TestCase):
    def test_auto_reply_keyword_detection(self) -> None:
        self.assertTrue(is_auto_reply_keyword("Thank you for contacting us"))
        self.assertTrue(is_auto_reply_keyword("Aapki jaankari ke liye shukriya"))
        self.assertFalse(is_auto_reply_keyword("Yes, go ahead"))

    def test_exact_opt_out(self) -> None:
        self.assertTrue(is_exact_opt_out("stop"))
        self.assertTrue(is_exact_opt_out("unsubscribe"))
        self.assertFalse(is_exact_opt_out("not interested"))

    def test_repeated_in_history(self) -> None:
        msg = "Thank you for contacting us"
        history = [
            {"from": "merchant", "body": msg},
            {"from": "bot", "body": "response"},
            {"from": "merchant", "body": msg},
            {"from": "merchant", "body": msg},
        ]
        self.assertTrue(repeated_in_history(msg, history, threshold=3))

    def test_keyword_fallback_commit(self) -> None:
        result = _keyword_fallback("ok go ahead", [])
        self.assertEqual(result["action"], "send")
        self.assertIn("commit", result["rationale"].lower())

    def test_keyword_fallback_hostile(self) -> None:
        result = _keyword_fallback("stop", [])
        self.assertEqual(result["action"], "end")

    def test_keyword_fallback_off_topic(self) -> None:
        result = _keyword_fallback("can you help with GST filing?", [])
        self.assertEqual(result["action"], "send")
        self.assertIn("off-topic", result["rationale"].lower())


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def test_health_and_metadata(self) -> None:
        self.assertEqual(routes.healthz()["status"], "ok")
        meta = routes.metadata()
        self.assertIn("compose + classify", meta["model"])
        self.assertEqual(meta["approach"], "hybrid-bm25-retrieval-plus-llm-composer")

    def test_health_accepts_head_for_monitors(self) -> None:
        methods = {
            method
            for route in app.routes
            if getattr(route, "path", "") == "/v1/healthz"
            for method in getattr(route, "methods", set())
        }
        self.assertIn("HEAD", methods)
        self.assertEqual(routes.healthz_head().status_code, 200)

    def test_context_stale_response_shape(self) -> None:
        context_id = "m-test-api"
        body = ContextBody(
            scope="merchant",
            context_id=context_id,
            version=2,
            payload={"merchant_id": context_id},
        )
        self.assertTrue(routes.push_context(body)["accepted"])
        stale = ContextBody(
            scope="merchant",
            context_id=context_id,
            version=1,
            payload={"merchant_id": "old"},
        )
        response = routes.push_context(stale)
        self.assertFalse(response["accepted"])
        self.assertEqual(response["reason"], "stale_version")


if __name__ == "__main__":
    unittest.main()
