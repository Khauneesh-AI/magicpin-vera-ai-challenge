from __future__ import annotations

from fastapi import FastAPI

from vera_bot.routes import router

app = FastAPI(title="Magicpin Vera Phase 1 Bot", version="1.0.0")
app.include_router(router, prefix="/v1")
