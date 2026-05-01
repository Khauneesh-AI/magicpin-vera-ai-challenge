from __future__ import annotations

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

    def add_turn(self, conversation_id: str, turn: dict[str, Any]) -> None:
        self.conversations.setdefault(conversation_id, []).append(turn)

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.conversations.setdefault(conversation_id, [])
