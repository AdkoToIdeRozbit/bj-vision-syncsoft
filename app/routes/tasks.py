import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.models import (
    BlackjackGameSession,
    BlackjackTrackingTask,
    BlackjackTrackingTaskStatus,
    GameResult,
)
from app.routes.deps import APIKeyDep, SessionDep
from app.vision import (
    build_vision_config,
    load_card_templates,
    map_card_names,
    map_player_hands,
    process_video,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"], dependencies=[APIKeyDep])

_active_job_id: int | None = None
_job_lock = threading.Lock()


class UploadVideoResponse(BaseModel):
    task_id: int


class TaskStatusResponse(BaseModel):
    task_id: str
    status: BlackjackTrackingTaskStatus
    csv_file_url: str | None = None
    game_sessions: list[GameResult] = []


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, session: SessionDep) -> TaskStatusResponse:
    task = session.get(BlackjackTrackingTask, task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )

    game_sessions = [GameResult.model_validate(gs.result) for gs in task.game_sessions]

    return TaskStatusResponse(
        task_id=task_id,
        status=task.status,
        csv_file_url=task.csv_output_path,
        game_sessions=game_sessions,
    )


# ---------------------------------------------------------------------------
# Background worker (runs in a thread-pool thread via FastAPI BackgroundTasks)
# ---------------------------------------------------------------------------


def _process_video_bg(task_id: int, video_path: str, profile: str) -> None:
    """Blocking vision processing job — runs in a threadpool thread.

    Opens its own DB session because the request session is already closed
    when this runs.
    """
    global _active_job_id

    with Session(engine) as session:
        task = session.get(BlackjackTrackingTask, task_id)
        if task is None:
            logger.error("Task %d not found in background worker", task_id)
            return

        try:
            # Mark as processing
            task.status = BlackjackTrackingTaskStatus.PROCESSING
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()

            # Build config and load templates
            vision_config = build_vision_config(
                video_path=video_path,
                profile=profile,  # type: ignore[arg-type]
                replay_threshold=settings.REPLAY_THRESHOLD,
                card_threshold=settings.CARD_THRESHOLD,
                cut_card_threshold=settings.CUT_CARD_THRESHOLD,
            )
            dealer_templates = load_card_templates(vision_config.dealer_template_dir)
            player_templates = load_card_templates(vision_config.player_template_dir)
            logger.info(
                "Task %d: loaded %d dealer / %d player templates",
                task_id,
                len(dealer_templates),
                len(player_templates),
            )

            # Run vision processing
            results = process_video(vision_config, dealer_templates, player_templates)
            logger.info(
                "[_process_video_bg] Task %d: detected %d session(s)",
                task_id,
                len(results),
            )

            # Persist each detected game session
            for r in results:
                game_result = GameResult(
                    session_number=r.get("session"),
                    deck_num=r.get("deck_num", 1),
                    dealer_cards=map_card_names(r.get("dealer", [])),
                    player1_cards=map_player_hands(r.get("player_1", {})),
                    player2_cards=map_player_hands(r.get("player_2", {})),
                    player3_cards=map_player_hands(r.get("player_3", {})),
                    player4_cards=map_player_hands(r.get("player_4", {})),
                    player5_cards=map_player_hands(r.get("player_5", {})),
                    player6_cards=map_player_hands(r.get("player_6", {})),
                    player7_cards=map_player_hands(r.get("player_7", {})),
                )
                game_session = BlackjackGameSession(
                    task_id=task_id,
                    result=game_result.model_dump(),
                )
                session.add(game_session)

            # Mark task as completed
            task.status = BlackjackTrackingTaskStatus.COMPLETED
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()

            logger.info("[_process_video_bg] Task %d completed successfully", task_id)

        except Exception:
            logger.exception("Task %d failed during vision processing", task_id)
            session.rollback()
            try:
                task.status = BlackjackTrackingTaskStatus.FAILED
                task.updated_at = datetime.now(timezone.utc)
                session.add(task)
                session.commit()
            except Exception:
                logger.exception("Task %d: could not update status to FAILED", task_id)

        finally:
            with _job_lock:
                _active_job_id = None
            # Remove the uploaded video file regardless of success or failure
            video_file = Path(video_path)
            if video_file.exists():
                try:
                    video_file.unlink()
                    logger.info("Task %d: deleted video file %s", task_id, video_path)
                except OSError:
                    logger.warning(
                        "Task %d: could not delete video file %s", task_id, video_path
                    )


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


@router.post("/upload", description="Upload a video for processing")
async def upload_video_for_processing(
    background_tasks: BackgroundTasks,
    session: SessionDep,
    file: UploadFile = File(...),
    profile: Literal["480p", "4k"] = Form("480p"),
) -> UploadVideoResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required"
        )

    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="content_type must be a video MIME type",
        )

    with _job_lock:
        global _active_job_id
        if _active_job_id is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A video is already being processed. Please wait!",
            )

        uploads_dir = Path(settings.LOCAL_UPLOAD_DIR)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        video_id = f"{uuid4().hex}-{file.filename}"
        storage_path = uploads_dir / video_id

        try:
            contents = await file.read()
            max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
            if len(contents) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"file is larger than {settings.MAX_UPLOAD_MB}MB limit",
                )
            storage_path.write_bytes(contents)
        finally:
            await file.close()

        # Create a new tracking task in the database
        task = BlackjackTrackingTask(
            video_path=str(storage_path), status=BlackjackTrackingTaskStatus.PENDING
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        # Reserve the active job slot
        _active_job_id = task.id

    # Trigger background processing
    background_tasks.add_task(
        _process_video_bg, task.id or 0, str(storage_path), profile
    )

    return UploadVideoResponse(task_id=task.id or 0)
