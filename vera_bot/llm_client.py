"""Async OpenAI client for composition and classification.

Uses the Responses API with Pydantic structured output via
client.responses.parse() + text_format.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from vera_bot import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------


class ComposedMessage(BaseModel):
    body: str
    cta: Literal["binary_yes_no", "open_ended", "none", "binary_confirm_cancel"]
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: str
    rationale: str


class ReplyClassification(BaseModel):
    intent: Literal[
        "auto_reply", "commit", "question", "engaged", "opt_out", "off_topic"
    ]
    action: Literal["send", "wait", "end"]


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Composition (tick pipeline)
# ---------------------------------------------------------------------------


async def compose_message(
    system_prompt: str,
    user_prompt: str,
) -> ComposedMessage:
    """Call GPT-5.4-nano to compose a merchant/customer message.

    Returns a typed ComposedMessage. Raises on timeout or API error —
    callers must handle fallback.
    """
    client = _get_client()
    response = await asyncio.wait_for(
        client.responses.parse(
            model=config.COMPOSE_MODEL,
            text_format=ComposedMessage,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.COMPOSE_TEMPERATURE,
            max_output_tokens=config.COMPOSE_MAX_TOKENS,
        ),
        timeout=config.COMPOSE_TIMEOUT,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise OpenAIError("LLM returned no parsed output")
    return parsed


# ---------------------------------------------------------------------------
# Reply classification
# ---------------------------------------------------------------------------


async def classify_reply(
    system_prompt: str,
    user_prompt: str,
) -> ReplyClassification:
    """Call GPT-5.4-mini to classify a merchant reply.

    Returns a typed ReplyClassification. Raises on timeout or API error.
    """
    client = _get_client()
    response = await asyncio.wait_for(
        client.responses.parse(
            model=config.CLASSIFY_MODEL,
            text_format=ReplyClassification,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.CLASSIFY_TEMPERATURE,
            max_output_tokens=config.CLASSIFY_MAX_TOKENS,
        ),
        timeout=config.CLASSIFY_TIMEOUT,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise OpenAIError("LLM returned no parsed output")
    return parsed


# ---------------------------------------------------------------------------
# Reply composition (when classification.action == "send")
# ---------------------------------------------------------------------------


async def compose_reply(
    system_prompt: str,
    user_prompt: str,
) -> ComposedMessage:
    """Call GPT-5.4-mini to compose a reply within an ongoing conversation.

    Same structured output as compose_message but uses the classify model
    (GPT-5.4-mini) for higher quality on multi-turn.
    """
    client = _get_client()
    response = await asyncio.wait_for(
        client.responses.parse(
            model=config.CLASSIFY_MODEL,
            text_format=ComposedMessage,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.COMPOSE_TEMPERATURE,
            max_output_tokens=config.COMPOSE_MAX_TOKENS,
        ),
        timeout=config.CLASSIFY_TIMEOUT,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise OpenAIError("LLM returned no parsed output")
    return parsed
