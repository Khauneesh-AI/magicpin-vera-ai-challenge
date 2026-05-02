from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Team
TEAM_NAME = "DakshTheCoder"
TEAM_MEMBERS: list[str] = ["Daksh Malhotra"]
CONTACT_EMAIL = "dakshmalhotra_23ep033@dtu.ac.in"
APPROACH = "hybrid-bm25-retrieval-plus-llm-composer"

# LLM Provider: "openai" or "gemini"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# OpenAI (fallback / selector)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

# Resolved model names based on provider
if LLM_PROVIDER == "gemini":
    COMPOSE_MODEL = GEMINI_MODEL
    CLASSIFY_MODEL = GEMINI_MODEL
    MODEL_NAME = f"{GEMINI_MODEL} (compose + classify)"
else:
    COMPOSE_MODEL = OPENAI_MODEL
    CLASSIFY_MODEL = OPENAI_MODEL
    MODEL_NAME = f"{OPENAI_MODEL} (compose + classify)"

COMPOSE_TEMPERATURE = 0.3
CLASSIFY_TEMPERATURE = 0.0
COMPOSE_TIMEOUT = 15.0 if LLM_PROVIDER == "gemini" else 8.0
CLASSIFY_TIMEOUT = 15.0 if LLM_PROVIDER == "gemini" else 5.0
COMPOSE_MAX_TOKENS = 500
CLASSIFY_MAX_TOKENS = 100

# Retrieval
BM25_CONFIDENCE_THRESHOLD = 3.5

# Message constraints
MAX_BODY_CHARS = 320
MAX_ACTIONS_PER_TICK = 20
ANTI_REPEAT_WINDOW = 5
ALLOWED_SEND_AS = {"vera", "merchant_on_behalf"}
