from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from vera_bot import config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def body_hash(body: str) -> str:
    return sha1(body.strip().lower().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-turn language detection
# ---------------------------------------------------------------------------

_HINDI_MARKERS = (
    "aapka", "aapke", "aapki", "hai", "hain", "kya", "kar", "karo",
    "ke liye", "ka kaam", "nahi", "haan", "chalega", "kar do",
    "se", "me", "mein", "ke", "ki", "ko", "pe", "bhi", "abhi",
    "shukriya", "dhanyavaad", "namaste", "ji",
)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")


def detect_language(text: str) -> str:
    """Detect if text is Hindi, English, or Hindi-English mix."""
    if _DEVANAGARI_RE.search(text):
        return "hi"
    lowered = text.lower()
    hindi_count = sum(1 for marker in _HINDI_MARKERS if marker in lowered)
    word_count = max(1, len(lowered.split()))
    if hindi_count >= 3 or hindi_count / word_count > 0.2:
        return "hi-en"
    return "en"


@dataclass
class StoredContext:
    version: int
    payload: dict[str, Any]
    stored_at: str


@dataclass
class Store:
    contexts: dict[tuple[str, str], StoredContext] = field(default_factory=dict)
    conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sent_suppressions: set[tuple[str, str]] = field(default_factory=set)
    recent_bodies: dict[str, deque[str]] = field(default_factory=dict)
    auto_reply_hashes: dict[str, list[str]] = field(default_factory=dict)
    # Unanswered nudge counter: merchant_id -> consecutive bot messages without merchant reply
    unanswered_nudges: dict[str, int] = field(default_factory=dict)
    # Cadence tracker: merchant_id -> list of send timestamps (ISO strings)
    send_cadence: dict[str, list[str]] = field(default_factory=dict)
    # Per-conversation detected language
    conversation_language: dict[str, str] = field(default_factory=dict)

    def push_context(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        key = (scope, context_id)
        current = self.contexts.get(key)
        if current and version < current.version:
            return {
                "accepted": False,
                "reason": "stale_version",
                "current_version": current.version,
            }
        if not current or version > current.version:
            self.contexts[key] = StoredContext(
                version=version,
                payload=payload,
                stored_at=utc_now_iso(),
            )
        stored = self.contexts[key]
        return {
            "accepted": True,
            "ack_id": f"ack_{context_id}_v{stored.version}",
            "stored_at": stored.stored_at,
        }

    def get_payload(self, scope: str, context_id: str | None) -> dict[str, Any] | None:
        if not context_id:
            return None
        stored = self.contexts.get((scope, context_id))
        return stored.payload if stored else None

    def count_contexts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for scope, _ in self.contexts:
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def mark_sent(self, merchant_id: str, suppression_key: str) -> None:
        self.sent_suppressions.add((merchant_id, suppression_key))

    def was_sent(self, merchant_id: str, suppression_key: str) -> bool:
        return (merchant_id, suppression_key) in self.sent_suppressions

    def remember_body(self, merchant_id: str, body: str) -> str:
        digest = body_hash(body)
        bodies = self.recent_bodies.setdefault(
            merchant_id,
            deque(maxlen=config.ANTI_REPEAT_WINDOW),
        )
        bodies.append(digest)
        return digest

    def seen_recent_body(self, merchant_id: str, body: str) -> bool:
        return body_hash(body) in self.recent_bodies.get(merchant_id, ())

    def remember_auto_reply(self, merchant_id: str, body: str) -> int:
        digest = body_hash(body)
        hashes = self.auto_reply_hashes.setdefault(merchant_id, [])
        hashes.append(digest)
        del hashes[:-10]
        return sum(1 for item in hashes if item == digest)

    def add_turn(self, conversation_id: str, turn: dict[str, Any]) -> None:
        self.conversations.setdefault(conversation_id, []).append(turn)

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.conversations.setdefault(conversation_id, [])

    # --- Unanswered nudge tracking ---

    def record_bot_send(self, merchant_id: str) -> None:
        """Increment unanswered nudge count for a merchant."""
        self.unanswered_nudges[merchant_id] = self.unanswered_nudges.get(merchant_id, 0) + 1

    def record_merchant_reply(self, merchant_id: str) -> None:
        """Reset unanswered nudge count when merchant replies."""
        self.unanswered_nudges[merchant_id] = 0

    def get_unanswered_count(self, merchant_id: str) -> int:
        return self.unanswered_nudges.get(merchant_id, 0)

    def should_stop_nudging(self, merchant_id: str) -> bool:
        """Return True if merchant has 3+ unanswered nudges."""
        return self.get_unanswered_count(merchant_id) >= 3

    # --- Cadence tracking ---

    def record_send_time(self, merchant_id: str) -> None:
        """Record that we sent a message to this merchant now."""
        times = self.send_cadence.setdefault(merchant_id, [])
        times.append(utc_now_iso())
        # Keep last 10 timestamps
        del times[:-10]

    def get_sends_in_window(self, merchant_id: str, window_hours: int = 24) -> int:
        """Count how many messages we sent to this merchant in the last N hours."""
        times = self.send_cadence.get(merchant_id, [])
        if not times:
            return 0
        now = datetime.now(timezone.utc)
        count = 0
        for ts in reversed(times):
            try:
                dt = datetime.fromisoformat(ts)
                if (now - dt).total_seconds() <= window_hours * 3600:
                    count += 1
                else:
                    break
            except (ValueError, TypeError):
                continue
        return count

    # --- Per-conversation language detection ---

    def detect_and_store_language(self, conversation_id: str, message: str) -> str:
        """Detect language of a message and store it for this conversation."""
        lang = detect_language(message)
        # Upgrade: if we detect Hindi in any turn, the conversation is hi-en
        current = self.conversation_language.get(conversation_id, "en")
        if lang in ("hi", "hi-en") or current in ("hi", "hi-en"):
            self.conversation_language[conversation_id] = "hi-en"
        else:
            self.conversation_language[conversation_id] = lang
        return self.conversation_language[conversation_id]

    def get_conversation_language(self, conversation_id: str) -> str:
        return self.conversation_language.get(conversation_id, "en")
