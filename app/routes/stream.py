import logging
from datetime import datetime, timezone
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.models import (
    BlackjackGameSession,
    BlackjackTrackingTask,
    BlackjackTrackingTaskStatus,
    GameResult,
)
from app.routes.deps import SessionDep
from app.vision import (
    LiveVideoProcessor,
    build_vision_config,
    load_card_templates,
    map_card_names,
    map_player_hands,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stream", tags=["stream"])


@router.websocket("")
async def stream_video_for_processing(
    websocket: WebSocket,
    session: SessionDep,
    api_key: str = "",
    profile: Literal["480p", "4k"] = "480p",
    lookback: int = 2,
):
    if api_key != settings.API_KEY:
        await websocket.close()
        return

    await websocket.accept()

    # Create a new tracking task in the database for this stream
    task = BlackjackTrackingTask(
        video_path="websocket-stream", status=BlackjackTrackingTaskStatus.PROCESSING
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    task_id = task.id or 0

    try:
        # Build config and load templates
        vision_config = build_vision_config(
            video_path="websocket-stream",
            profile=profile,  # type: ignore[arg-type]
            replay_threshold=settings.REPLAY_THRESHOLD,
            card_threshold=settings.CARD_THRESHOLD,
            lookback=lookback,
            cut_card_threshold=settings.CUT_CARD_THRESHOLD,
        )
        dealer_templates = load_card_templates(vision_config.dealer_template_dir)
        player_templates = load_card_templates(vision_config.player_template_dir)
        logger.info(
            "Stream Task %d: loaded %d dealer / %d player templates",
            task_id,
            len(dealer_templates),
            len(player_templates),
        )

        processor = LiveVideoProcessor(
            vision_config, dealer_templates, player_templates
        )

        await websocket.send_json(
            {"type": "status", "task_id": task_id, "status": "processing"}
        )

        while True:
            # We expect bytes of an encoded image (e.g. JPEG)
            data = await websocket.receive_bytes()

            # Decode the bytes directly to an OpenCV frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning("Stream Task %d: Failed to decode frame", task_id)
                continue

            # Process frame using a thread pool to avoid blocking the async event loop
            result = await run_in_threadpool(processor.process_frame, frame)

            if result is not None:
                # Save to database
                game_result = GameResult(
                    session_number=result.get("session"),
                    deck_num=result.get("deck_num", 1),
                    dealer_cards=map_card_names(result.get("dealer", [])),
                    player1_cards=map_player_hands(result.get("player_1", {})),
                    player2_cards=map_player_hands(result.get("player_2", {})),
                    player3_cards=map_player_hands(result.get("player_3", {})),
                    player4_cards=map_player_hands(result.get("player_4", {})),
                    player5_cards=map_player_hands(result.get("player_5", {})),
                    player6_cards=map_player_hands(result.get("player_6", {})),
                    player7_cards=map_player_hands(result.get("player_7", {})),
                )
                game_session = BlackjackGameSession(
                    task_id=task_id,
                    result=game_result.model_dump(),
                )
                session.add(game_session)
                session.commit()

                # Emit to client immediately
                await websocket.send_json(
                    {
                        "type": "result",
                        "session_number": result.get("session"),
                        "frame_index": result.get("frame"),
                        "data": game_result.model_dump(),
                    }
                )

    except WebSocketDisconnect:
        logger.info("Stream Task %d: WebSocket disconnected", task_id)
    except Exception as e:
        logger.exception(
            "Stream Task %d failed during vision processing: %s", task_id, e
        )
        try:
            task.status = BlackjackTrackingTaskStatus.FAILED
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()
        except:  # noqa: E722
            pass
    finally:
        # If not failed, mark as completed
        try:
            session.refresh(task)
            if task.status != BlackjackTrackingTaskStatus.FAILED:
                task.status = BlackjackTrackingTaskStatus.COMPLETED
                task.updated_at = datetime.now(timezone.utc)
                session.add(task)
                session.commit()
        except Exception:
            pass
