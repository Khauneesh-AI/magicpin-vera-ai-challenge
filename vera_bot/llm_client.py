"""Async LLM client for composition and classification.

Supports two providers:
- OpenAI: uses Responses API with client.responses.parse() + text_format
- Gemini: uses native google-genai SDK with JSON schema output
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from vera_bot import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

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
# OpenAI client
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


async def _gemini_parse(
    model: str,
    schema: type[T],
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> T:
    """Call Gemini via native google-genai SDK with JSON schema output."""
    from google.genai import types

    client = _get_gemini_client()

    # Convert OpenAI-style messages to Gemini format
    system_text = ""
    user_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            user_text = msg["content"]

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_text)],
    )]

    generate_config = types.GenerateContentConfig(
        system_instruction=system_text if system_text else None,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # Run sync generate_content in a thread to not block the event loop
    loop = asyncio.get_event_loop()
    response = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_config,
            ),
        ),
        timeout=timeout,
    )

    text = response.text
    if not text:
        raise ValueError("Gemini returned empty response")

    data = json.loads(text)
    return schema.model_validate(data)


# ---------------------------------------------------------------------------
# Unified parse — picks provider
# ---------------------------------------------------------------------------


async def _parse(
    model: str,
    schema: type[T],
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> T:
    if config.LLM_PROVIDER == "gemini":
        return await _gemini_parse(model, schema, messages, temperature, timeout)
    else:
        from openai import OpenAIError
        client = _get_openai_client()
        response = await asyncio.wait_for(
            client.responses.parse(
                model=model,
                text_format=schema,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            timeout=timeout,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise OpenAIError("LLM returned no parsed output")
        return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compose_message(
    system_prompt: str,
    user_prompt: str,
) -> ComposedMessage:
    """Compose a merchant/customer message."""
    return await _parse(
        model=config.COMPOSE_MODEL,
        schema=ComposedMessage,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.COMPOSE_TEMPERATURE,
        max_tokens=config.COMPOSE_MAX_TOKENS,
        timeout=config.COMPOSE_TIMEOUT,
    )


async def classify_reply(
    system_prompt: str,
    user_prompt: str,
) -> ReplyClassification:
    """Classify a merchant reply."""
    return await _parse(
        model=config.CLASSIFY_MODEL,
        schema=ReplyClassification,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.CLASSIFY_TEMPERATURE,
        max_tokens=config.CLASSIFY_MAX_TOKENS,
        timeout=config.CLASSIFY_TIMEOUT,
    )


async def compose_reply(
    system_prompt: str,
    user_prompt: str,
) -> ComposedMessage:
    """Compose a reply within an ongoing conversation."""
    return await _parse(
        model=config.CLASSIFY_MODEL,
        schema=ComposedMessage,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.COMPOSE_TEMPERATURE,
        max_tokens=config.COMPOSE_MAX_TOKENS,
        timeout=config.CLASSIFY_TIMEOUT,
    )
