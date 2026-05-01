from __future__ import annotations

import json
import unittest
from pathlib import Path

from vera_bot.composer import compose
from tools.generate_submission import build_rows
from vera_bot.main import app
from tools.validate_submission import REQUIRED, URL_RE, validate_rows
from vera_bot import routes
from vera_bot.reply_handler import decide_reply
from vera_bot.schemas import ContextBody, ReplyBody
from vera_bot.store import Store
from vera_bot.validators import finalize_message

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "expanded"


def load_context(scope: str, context_id: str) -> dict:
    folder = {
        "category": "categories",
        "merchant": "merchants",
        "trigger": "triggers",
    }[scope]
    return json.loads((EXPANDED / folder / f"{context_id}.json").read_text(encoding="utf-8"))


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


class ApiTests(unittest.TestCase):
    def test_health_and_metadata(self) -> None:
        self.assertEqual(routes.healthz()["status"], "ok")
        metadata = routes.metadata()
        self.assertEqual(metadata["model"], "deterministic-no-llm")

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


class CustomerFallbackQualityTests(unittest.TestCase):
    def compose_without_customer(self, trigger_id: str) -> dict[str, str]:
        trigger = load_context("trigger", trigger_id)
        merchant = load_context("merchant", trigger["merchant_id"])
        category = load_context("category", merchant["category_slug"])
        return compose(category, merchant, trigger, customer=None)

    def assert_payload_aware_customer_fallback(
        self,
        trigger_id: str,
        required_terms: tuple[str, ...],
    ) -> None:
        msg = self.compose_without_customer(trigger_id)
        body = msg["body"].lower()
        self.assertNotIn("customer trigger", body)
        self.assertEqual(msg["send_as"], "merchant_on_behalf")
        for term in required_terms:
            self.assertIn(term.lower(), body)

    def test_recall_fallback_uses_service_and_slot(self) -> None:
        self.assert_payload_aware_customer_fallback(
            "trg_003_recall_due_priya",
            ("6-month cleaning", "Wed 5 Nov"),
        )

    def test_bridal_fallback_uses_wedding_countdown(self) -> None:
        self.assert_payload_aware_customer_fallback(
            "trg_007_bridal_followup_kavya",
            ("196", "skin prep"),
        )

    def test_gym_lapsed_fallback_uses_days_and_focus(self) -> None:
        self.assert_payload_aware_customer_fallback(
            "trg_015_winback_rashmi",
            ("57 days", "weight loss"),
        )

    def test_trial_fallback_uses_trial_and_next_slot(self) -> None:
        self.assert_payload_aware_customer_fallback(
            "trg_017_kids_yoga_trial_followup_karthik",
            ("2026-04-22", "Sat 3 May"),
        )

    def test_refill_fallback_uses_meds_and_runout_date(self) -> None:
        self.assert_payload_aware_customer_fallback(
            "trg_019_chronic_refill_grandfather",
            ("metformin", "2026-04-28"),
        )


class GeneralizationQualityTests(unittest.TestCase):
    def test_unknown_merchant_trigger_summarizes_payload_and_context(self) -> None:
        category = {
            "slug": "restaurants",
            "display_name": "Restaurants",
        }
        merchant = {
            "merchant_id": "m_synthetic_cafe",
            "category_slug": "restaurants",
            "identity": {
                "name": "Test Cafe",
                "owner_first_name": "Asha",
                "locality": "Indiranagar",
                "city": "Bangalore",
            },
            "performance": {"views": 4200, "calls": 31, "ctr": 0.034},
            "offers": [{"title": "Lunch Combo @ Rs199", "status": "active"}],
        }
        trigger = {
            "id": "trg_synthetic",
            "kind": "new_local_signal",
            "payload": {
                "search_query": "family lunch near me",
                "delta_yoy": 0.41,
                "window": "7d",
            },
        }
        msg = compose(category, merchant, trigger)
        body = msg["body"].lower()
        for term in ("asha", "indiranagar", "4200", "31", "lunch combo", "family lunch near me", "41%"):
            self.assertIn(term, body)
        self.assertNotIn("metric_or_topic", body)

    def test_curious_ask_uses_review_or_signal_when_available(self) -> None:
        category = {"slug": "salons", "display_name": "Salons"}
        merchant = {
            "merchant_id": "m_synthetic_salon",
            "category_slug": "salons",
            "identity": {"name": "Glow Room", "owner_first_name": "Naina"},
            "review_themes": [
                {"theme": "hair spa", "sentiment": "pos", "occurrences_30d": 9},
            ],
            "signals": ["weekday_afternoon_gap"],
        }
        trigger = {"id": "ask_synthetic", "kind": "curious_ask_due", "payload": {}}
        msg = compose(category, merchant, trigger)
        body = msg["body"].lower()
        self.assertIn("hair spa", body)
        self.assertIn("9", body)
        self.assertIn("google post", body)


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
        self.assertIn("commit", decision["rationale"].lower())

    def test_repeated_auto_reply_ends_across_conversations_for_same_merchant(self) -> None:
        original_store = routes.store
        routes.store = Store()
        try:
            actions = []
            for index in range(1, 5):
                decision = routes.reply(
                    ReplyBody(
                        conversation_id=f"auto-repeat-{index}",
                        merchant_id="merchant-auto-repeat",
                        message="Thank you for contacting us! Our team will respond shortly.",
                        turn_number=index,
                    )
                )
                actions.append(decision["action"])
            self.assertIn("end", actions)
        finally:
            routes.store = original_store


if __name__ == "__main__":
    unittest.main()
