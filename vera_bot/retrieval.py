"""BM25 retrieval over the templates corpus.

Loads templates_corpus.json at import time and exposes a single query
function used by the composer to find the top-N most relevant template
examples for a given (trigger, category) pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_CORPUS_PATH = Path(__file__).with_name("templates_corpus.json")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_doc(entry: dict[str, Any]) -> str:
    parts = [
        entry.get("trigger_kind", ""),
        entry.get("category", ""),
        entry.get("scope", ""),
        entry.get("cta_type", ""),
        " ".join(entry.get("facts_used", [])),
        " ".join(entry.get("compulsion_levers", [])),
    ]
    return " ".join(parts)


class TemplateIndex:
    def __init__(self, corpus_path: Path = _CORPUS_PATH) -> None:
        raw = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.entries: list[dict[str, Any]] = raw
        tokenized = [_tokenize(_build_doc(e)) for e in self.entries]
        self.bm25 = BM25Okapi(tokenized)

    def query(
        self,
        trigger: dict[str, Any],
        category: dict[str, Any],
        top_n: int = 2,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return top-N matches as (entry, score) tuples."""
        kind = trigger.get("kind", "")
        slug = category.get("slug", "")
        payload_keys = " ".join(list(trigger.get("payload", {}).keys())[:5])
        scope = trigger.get("scope", "merchant")
        query_str = f"{kind} {slug} {scope} {payload_keys}"
        tokens = _tokenize(query_str)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.entries)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [(self.entries[i], scores[i]) for i in ranked[:top_n]]


# Module-level singleton — built once at import time.
index = TemplateIndex()
