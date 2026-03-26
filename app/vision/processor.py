"""
Core vision processing module for blackjack card detection.

Ported from scripts/pattern-matching.py — CLI and env-loading removed.
Frame saving removed — results are persisted to the database by the caller.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app.models import CardClass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VISION_BASE = Path(__file__).parent / "template-images"


@dataclass
class VisionConfig:
    video_path: str

    replay_template: str
    replay_roi: tuple[int, int, int, int] | None
    replay_threshold: float

    dealer_template_dir: str
    player_template_dir: str
    dealer_roi: tuple[int, int, int, int] | None
    # keys: "player_1" … "player_7", value None means skip
    player_rois: dict[str, tuple[int, int, int, int] | None] = field(
        default_factory=dict
    )

    card_threshold: float = 0.8


# Hard-coded ROI defaults per resolution profile.
# These come from the .env comments; callers can override thresholds via
# build_vision_config().
_PROFILE_DEFAULTS: dict[str, dict] = {
    "480p": {
        "replay_template": str(_VISION_BASE / "480p" / "replay.jpg"),
        "replay_roi": (220, 223, 240, 240),
        "dealer_template_dir": str(_VISION_BASE / "480p" / "dealer"),
        "player_template_dir": str(_VISION_BASE / "480p" / "players"),
        "dealer_roi": (230, 83, 290, 91),
        "player_rois": {
            "player_1": None,
            "player_2": None,
            "player_3": None,
            "player_4": (399, 290, 407, 374),
            "player_5": None,
            "player_6": (525, 260, 533, 358),
            "player_7": None,
        },
    },
    "4k": {
        "replay_template": str(_VISION_BASE / "4k" / "replay.jpg"),
        "replay_roi": (1000, 1020, 1150, 1150),
        "dealer_template_dir": str(_VISION_BASE / "4k" / "dealer"),
        "player_template_dir": str(_VISION_BASE / "4k" / "players"),
        "dealer_roi": (1070, 560, 1300, 593),
        "player_rois": {
            "player_1": (1045, 1183, 1077, 1467),
            "player_2": (1225, 1260, 1255, 1534),
            "player_3": (1440, 1300, 1475, 1580),
            "player_4": (1675, 1240, 1705, 1594),
            "player_5": (1910, 1300, 1938, 1578),
            "player_6": (2126, 1270, 2156, 1536),
            "player_7": (2305, 1200, 2335, 1464),
        },
    },
}

ProfileName = Literal["480p", "4k"]


def build_vision_config(
    video_path: str,
    profile: ProfileName,
    replay_threshold: float = 0.8,
    card_threshold: float = 0.8,
) -> VisionConfig:
    """Build a VisionConfig for the given resolution profile."""
    defaults = _PROFILE_DEFAULTS[profile]
    return VisionConfig(
        video_path=video_path,
        replay_template=defaults["replay_template"],
        replay_roi=defaults["replay_roi"],
        replay_threshold=replay_threshold,
        dealer_template_dir=defaults["dealer_template_dir"],
        player_template_dir=defaults["player_template_dir"],
        dealer_roi=defaults["dealer_roi"],
        player_rois=defaults["player_rois"],
        card_threshold=card_threshold,
    )


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def load_card_templates(directory: str) -> list[tuple[str, np.ndarray]]:
    """Load all image files from *directory* as grayscale templates.

    Returns a list of ``(card_name, grayscale_image)`` tuples where
    *card_name* is the filename stem (e.g. ``"ace"`` for ``ace.png``).
    """
    templates: list[tuple[str, np.ndarray]] = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Template directory not found: {directory}")

    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((file.stem, img))

    if not templates:
        logger.warning("No templates found in %s", directory)
    return templates


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _merge_detections(
    points: list[tuple[int, int]], template_w: int, template_h: int
) -> int:
    """Merge overlapping detection points into distinct clusters.

    Points within one template-width/height distance belong to the same
    physical card. Returns the number of distinct card instances.
    """
    if not points:
        return 0

    points.sort()
    clusters: list[tuple[int, int]] = []

    for px, py in points:
        merged = False
        for i, (cx, cy) in enumerate(clusters):
            if abs(px - cx) < template_w and abs(py - cy) < template_h:
                clusters[i] = ((cx + px) // 2, (cy + py) // 2)
                merged = True
                break
        if not merged:
            clusters.append((px, py))

    return len(clusters)


def detect_cards_in_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    templates: list[tuple[str, np.ndarray]],
    threshold: float,
) -> list[str]:
    """Match all templates against a ROI in the frame.

    Groups overlapping detections of the same template into distinct card
    instances. Returns a list of detected card name stems (may contain
    duplicates for multiple physical cards of the same rank).
    """
    x1, y1, x2, y2 = roi
    region = frame[y1:y2, x1:x2]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    detected_cards: list[str] = []

    for card_name, template in templates:
        th, tw = template.shape[:2]

        if tw > gray_region.shape[1] or th > gray_region.shape[0]:
            continue

        result = cv2.matchTemplate(gray_region, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        if len(locations[0]) == 0:
            continue

        points = list(zip(locations[1].tolist(), locations[0].tolist()))
        merged_count = _merge_detections(points, tw, th)
        for _ in range(merged_count):
            detected_cards.append(card_name)

    return detected_cards


def _detect_cards_for_session(
    frame_idx: int,
    frame: np.ndarray,
    config: VisionConfig,
    dealer_templates: list[tuple[str, np.ndarray]],
    player_templates: list[tuple[str, np.ndarray]],
) -> dict | None:
    """Detect dealer and player cards in a single session-ending frame.

    Returns a result dict, or ``None`` if no dealer cards were detected
    (treated as a false-positive replay trigger).
    """
    session_result: dict = {"frame": frame_idx}

    # Dealer cards
    if config.dealer_roi is not None:
        dealer_cards = detect_cards_in_roi(
            frame, config.dealer_roi, dealer_templates, config.card_threshold
        )
        session_result["dealer"] = dealer_cards

        if not dealer_cards:
            # Likely a false replay detection — skip this session
            return None

        logger.debug("Dealer cards: %s", dealer_cards)
    else:
        session_result["dealer"] = []
        logger.debug("Dealer ROI not configured, skipping")

    # Player cards (1–7)
    for i in range(1, 8):
        key = f"player_{i}"
        player_roi = config.player_rois.get(key)
        if player_roi is not None:
            player_cards = detect_cards_in_roi(
                frame, player_roi, player_templates, config.card_threshold
            )
            session_result[key] = player_cards
            logger.debug("Player %d cards: %s", i, player_cards)
        else:
            session_result[key] = []

    return session_result


# ---------------------------------------------------------------------------
# Main video processing
# ---------------------------------------------------------------------------


def process_video(
    config: VisionConfig,
    dealer_templates: list[tuple[str, np.ndarray]],
    player_templates: list[tuple[str, np.ndarray]],
) -> list[dict]:
    """Scan *video_path* for replay-button appearances and detect cards inline.

    For each session-ending frame detected, card detection runs immediately so
    results are produced in a single pass.

    Returns a list of session result dicts, each containing:
    - ``"session"``: 1-based session number
    - ``"frame"``: frame index of the session-ending frame
    - ``"dealer"``: list of detected dealer card name stems
    - ``"player_1"`` … ``"player_7"``: list of detected card name stems (empty if ROI not configured)
    """
    replay_template_img = cv2.imread(config.replay_template, cv2.IMREAD_GRAYSCALE)
    if replay_template_img is None:
        raise FileNotFoundError(
            f"Could not read replay template: {config.replay_template}"
        )

    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config.video_path}")

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if config.replay_roi is not None:
        x1, y1, x2, y2 = config.replay_roi
        if x2 > frame_width or y2 > frame_height:
            raise ValueError(
                f"Replay ROI ({x1},{y1},{x2},{y2}) exceeds frame dimensions "
                f"({frame_width}x{frame_height})"
            )

    logger.info(
        "Processing video: %s  [%dx%d]  replay_threshold=%.2f  card_threshold=%.2f",
        config.video_path,
        frame_width,
        frame_height,
        config.replay_threshold,
        config.card_threshold,
    )

    LOOKBACK = 5
    buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=LOOKBACK + 1)
    results: list[dict] = []

    frame_index = 0
    session_num = 1
    replay_active = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        buffer.append((frame_index, frame.copy()))

        if config.replay_roi is not None:
            x1, y1, x2, y2 = config.replay_roi
            search_region = frame[y1:y2, x1:x2]
        else:
            search_region = frame

        gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        match_result = cv2.matchTemplate(
            gray_region, replay_template_img, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, _ = cv2.minMaxLoc(match_result)

        if max_val >= config.replay_threshold and not replay_active:
            replay_active = True
            target_idx, target_frame = buffer[0]

            session_result = _detect_cards_for_session(
                target_idx,
                target_frame,
                config,
                dealer_templates,
                player_templates,
            )

            if session_result is not None:
                logger.info(
                    "Session %d: replay at frame %d, cards detected at frame %d",
                    session_num,
                    frame_index,
                    target_idx,
                )
                results.append({"session": session_num, **session_result})
                session_num = len(results) + 1

        elif max_val < config.replay_threshold and replay_active:
            replay_active = False

        frame_index += 1

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Card name mapping
# ---------------------------------------------------------------------------


def map_card_names(names: list[str]) -> list[CardClass]:
    """Map template stem names to ``CardClass`` enum values.

    Unknown names fall back to ``CardClass.UNKNOWN``.
    """
    result: list[CardClass] = []
    for name in names:
        try:
            result.append(CardClass(name.lower()))
        except ValueError:
            logger.warning("Unrecognised card name %r, mapping to UNKNOWN", name)
            result.append(CardClass.UNKNOWN)
    return result
