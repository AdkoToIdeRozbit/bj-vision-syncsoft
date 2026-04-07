import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.logging import ColourizedFormatter

from app.core.config import settings
from app.routes.game_sessions import router as game_sessions_router
from app.routes.stream import router as stream_router
from app.routes.tasks import router as tasks_router
from app.routes.webrtc import close_all_peer_connections
from app.routes.webrtc import router as webrtc_router

# Attach a colourized handler directly to the "app" logger so all app.*
# sub-loggers inherit it. propagate=False prevents records from also
# travelling up to the root logger and being printed a second time by
# uvicorn's handlers.
_handler = logging.StreamHandler()
_handler.setFormatter(
    ColourizedFormatter(fmt="%(levelprefix)s %(name)s — %(message)s", use_colors=True)
)
_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.DEBUG)
_app_logger.addHandler(_handler)
_app_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_all_peer_connections()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks_router)
app.include_router(stream_router)
app.include_router(game_sessions_router)
app.include_router(webrtc_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
