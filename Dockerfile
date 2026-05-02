FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY bot.py ./
COPY vera_bot/ ./vera_bot/
COPY expanded/ ./expanded/
COPY dataset/ ./dataset/

# Port from environment (Railway sets $PORT)
ENV PORT=8080
EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn bot:app --host 0.0.0.0 --port ${PORT}"]
