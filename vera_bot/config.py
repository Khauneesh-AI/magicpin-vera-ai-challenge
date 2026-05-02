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
MODEL_NAME = "gpt-5.4-mini (compose + classify)"

# LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COMPOSE_MODEL = "gpt-5.4-mini"
CLASSIFY_MODEL = "gpt-5.4-mini"
COMPOSE_TEMPERATURE = 0.3
CLASSIFY_TEMPERATURE = 0.0
COMPOSE_TIMEOUT = 8.0
CLASSIFY_TIMEOUT = 5.0
COMPOSE_MAX_TOKENS = 500
CLASSIFY_MAX_TOKENS = 100

# Retrieval
BM25_CONFIDENCE_THRESHOLD = 3.5  # above = use deterministic template, below = full LLM compose

# Message constraints
MAX_BODY_CHARS = 320
MAX_ACTIONS_PER_TICK = 20
ANTI_REPEAT_WINDOW = 5
ALLOWED_SEND_AS = {"vera", "merchant_on_behalf"}
