from __future__ import annotations

import logging

from fastapi import FastAPI

from vera_bot.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Magicpin Vera Phase 2 Bot", version="2.0.0")
app.include_router(router, prefix="/v1")
