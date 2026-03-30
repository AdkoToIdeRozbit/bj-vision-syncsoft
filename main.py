import logging

from fastapi import FastAPI
from uvicorn.logging import ColourizedFormatter

from app.core.config import settings
from app.routes.game_sessions import router as game_sessions_router
from app.routes.stream import router as stream_router
from app.routes.tasks import router as tasks_router

# Attach a colourized handler directly to the "app" logger so all app.*
# sub-loggers inherit it. propagate=False prevents records from also
# travelling up to the root logger and being printed a second time by
# uvicorn's handlers.
_handler = logging.StreamHandler()
_handler.setFormatter(
    ColourizedFormatter(fmt="%(levelprefix)s %(name)s — %(message)s", use_colors=True)
)
_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)
_app_logger.addHandler(_handler)
_app_logger.propagate = False

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(tasks_router)
app.include_router(stream_router)
app.include_router(game_sessions_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
