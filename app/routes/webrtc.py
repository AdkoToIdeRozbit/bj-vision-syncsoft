import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

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

router = APIRouter(prefix="/api/webrtc", tags=["webrtc"])

# Tracks all active peer connections for graceful shutdown
_pcs: set[RTCPeerConnection] = set()
_relay = MediaRelay()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    profile: Literal["480p", "4k"] = "480p"
    lookback: int = 2


class WebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
    task_id: int


# ---------------------------------------------------------------------------
# Video processing track
# ---------------------------------------------------------------------------


class _VideoProcessingTrack(MediaStreamTrack):
    """Receives video frames from the browser, feeds them into LiveVideoProcessor,
    and enqueues detection results onto a shared asyncio queue."""

    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        processor: LiveVideoProcessor,
        result_queue: asyncio.Queue,
        task_id: int = 0,
    ) -> None:
        super().__init__()
        self._track = track
        self._processor = processor
        self._result_queue = result_queue
        self._task_id = task_id
        self._frame_count = 0
        self._result_count = 0

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        frame = await self._track.recv()

        # Convert PyAV VideoFrame → OpenCV BGR ndarray
        img: np.ndarray = frame.to_ndarray(format="bgr24")  # type: ignore[union-attr]

        self._frame_count += 1

        # Log frame dimensions on the very first frame so we can verify resolution
        if self._frame_count == 1:
            h, w = img.shape[:2]
            logger.info(
                "WebRTC Task %d: first frame received — %dx%d (w×h)",
                self._task_id,
                w,
                h,
            )

        # Periodic progress log every 150 frames
        if self._frame_count % 150 == 0:
            logger.info(
                "WebRTC Task %d: processed %d frames, %d detections so far",
                self._task_id,
                self._frame_count,
                self._result_count,
            )

        # Run blocking CV2 processing off the event loop
        result = await run_in_threadpool(self._processor.process_frame, img)

        if result is not None:
            self._result_count += 1
            logger.info(
                "WebRTC Task %d: detection #%d at frame %d (session %s)",
                self._task_id,
                self._result_count,
                self._frame_count,
                result.get("session"),
            )
            await self._result_queue.put(result)

        # Return the original frame unchanged (we only care about server-side detection)
        return frame  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Signaling endpoint
# ---------------------------------------------------------------------------


@router.post("/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(
    body: WebRTCOfferRequest,
    session: SessionDep,
    api_key: str = "",
) -> WebRTCOfferResponse:
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Create a tracking task in the database
    task = BlackjackTrackingTask(
        video_path="webrtc-stream",
        status=BlackjackTrackingTaskStatus.PROCESSING,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    task_id: int = task.id or 0

    # Build vision config and load templates
    vision_config = build_vision_config(
        video_path="webrtc-stream",
        profile=body.profile,  # type: ignore[arg-type]
        replay_threshold=settings.REPLAY_THRESHOLD,
        card_threshold=settings.CARD_THRESHOLD,
        lookback=body.lookback,
    )
    dealer_templates = load_card_templates(vision_config.dealer_template_dir)
    player_templates = load_card_templates(vision_config.player_template_dir)
    logger.info(
        "WebRTC Task %d: loaded %d dealer / %d player templates",
        task_id,
        len(dealer_templates),
        len(player_templates),
    )

    processor = LiveVideoProcessor(vision_config, dealer_templates, player_templates)

    # Queue used to pass detection results from the processing track to the data-channel sender
    result_queue: asyncio.Queue = asyncio.Queue()

    # RTCPeerConnection for this session
    pc = RTCPeerConnection()
    _pcs.add(pc)

    # Holds the results data channel once the browser connects it
    data_channel_ref: list = []  # mutable container so inner async closures can write

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        state = pc.connectionState
        logger.info("WebRTC Task %d: connection state → %s", task_id, state)
        if state in ("failed", "closed"):
            await pc.close()
            _pcs.discard(pc)
            # Mark task as completed/failed in a fresh DB access
            # (the original `session` is already closed after the request returns)
            _finish_task_in_background(task_id, state)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-untyped-def]
        logger.info("WebRTC Task %d: data channel '%s' opened", task_id, channel.label)
        data_channel_ref.append(channel)

        # Kick off the coroutine that drains the result queue and sends over this channel
        asyncio.ensure_future(_drain_results(channel, result_queue, task_id))

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-untyped-def]
        logger.info("WebRTC Task %d: received %s track", task_id, track.kind)
        if track.kind == "video":
            processing_track = _VideoProcessingTrack(
                _relay.subscribe(track), processor, result_queue, task_id
            )
            # Adding the track to the peer connection keeps it alive and triggers recv() calls
            pc.addTrack(processing_track)

    # SDP offer/answer exchange
    offer = RTCSessionDescription(sdp=body.sdp, type=body.type)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return WebRTCOfferResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------


async def _drain_results(
    channel,  # RTCDataChannel
    result_queue: asyncio.Queue,
    task_id: int,
) -> None:
    """Continuously pull results from the queue and send them over the data channel."""
    while True:
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            # No result for 60 s — channel may still be alive; keep waiting
            continue
        except asyncio.CancelledError:
            break

        try:
            game_result = GameResult(
                session_number=result.get("session"),
                dealer_cards=map_card_names(result.get("dealer", [])),
                player1_cards=map_player_hands(result.get("player_1", {})),
                player2_cards=map_player_hands(result.get("player_2", {})),
                player3_cards=map_player_hands(result.get("player_3", {})),
                player4_cards=map_player_hands(result.get("player_4", {})),
                player5_cards=map_player_hands(result.get("player_5", {})),
                player6_cards=map_player_hands(result.get("player_6", {})),
                player7_cards=map_player_hands(result.get("player_7", {})),
            )

            # Persist to DB
            from sqlmodel import Session as DBSession

            from app.core.db import engine

            with DBSession(engine) as db:
                game_session = BlackjackGameSession(
                    task_id=task_id,
                    result=game_result.model_dump(),
                )
                db.add(game_session)
                db.commit()

            import json

            payload = json.dumps(
                {
                    "type": "result",
                    "task_id": task_id,
                    "session_number": result.get("session"),
                    "frame_index": result.get("frame"),
                    "data": game_result.model_dump(),
                }
            )
            if channel.readyState == "open":
                channel.send(payload)
        except Exception as exc:
            logger.exception("WebRTC Task %d: error sending result: %s", task_id, exc)


def _finish_task_in_background(task_id: int, state: str) -> None:
    """Schedule a fire-and-forget coroutine to mark the task finished."""
    asyncio.ensure_future(_finish_task(task_id, state))


async def _finish_task(task_id: int, state: str) -> None:
    try:
        from sqlmodel import Session as DBSession

        from app.core.db import engine

        final_status = (
            BlackjackTrackingTaskStatus.FAILED
            if state == "failed"
            else BlackjackTrackingTaskStatus.COMPLETED
        )
        with DBSession(engine) as db:
            task = db.get(BlackjackTrackingTask, task_id)
            if task and task.status == BlackjackTrackingTaskStatus.PROCESSING:
                task.status = final_status
                task.updated_at = datetime.now(timezone.utc)
                db.add(task)
                db.commit()
        logger.info("WebRTC Task %d: marked %s", task_id, final_status)
    except Exception as exc:
        logger.exception("WebRTC Task %d: failed to update status: %s", task_id, exc)


# ---------------------------------------------------------------------------
# Shutdown helper (called from app lifespan)
# ---------------------------------------------------------------------------


async def close_all_peer_connections() -> None:
    """Gracefully close every active RTCPeerConnection. Call on app shutdown."""
    coros = [pc.close() for pc in list(_pcs)]
    await asyncio.gather(*coros, return_exceptions=True)
    _pcs.clear()
