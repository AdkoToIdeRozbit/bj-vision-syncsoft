"""
Core vision processing module for blackjack card detection.

Ported from scripts/pattern-matching.py — CLI and env-loading removed.
Frame saving removed — results are persisted to the database by the caller.
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
class PlayerROIConfig:
    """ROI configuration for a single player, supporting split-hand detection.

    ``default`` covers the player's normal (un-split) hand position.
    ``split1`` and ``split2`` are the two hand positions after a split.
    Detection short-circuits: if ``default`` yields cards the split ROIs are
    not checked; otherwise ``split1`` is tried and, if cards are found there,
    ``split2`` is checked next.
    """

    default: tuple[int, int, int, int] | None = None
    split1: tuple[int, int, int, int] | None = None
    split2: tuple[int, int, int, int] | None = None


@dataclass
class VisionConfig:
    video_path: str

    replay_template: str
    replay_roi: tuple[int, int, int, int] | None
    replay_threshold: float

    dealer_template_dir: str
    player_template_dir: str
    dealer_roi: tuple[int, int, int, int] | None
    # keys: "player_1" … "player_7", None means skip that player entirely
    player_rois: dict[str, PlayerROIConfig | None] = field(default_factory=dict)

    card_threshold: float = 0.8
    lookback: int = 5

    cut_card_roi: tuple[int, int, int, int] | None = None
    cut_card_threshold: float = 0.15


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
        # TODO: Set the actual cut-card ROI coordinates for 480p footage
        "cut_card_roi": None,
        "player_rois": {
            "player_1": None,
            "player_2": None,
            "player_3": None,
            "player_4": PlayerROIConfig(default=(399, 290, 407, 374)),
            "player_5": None,
            "player_6": PlayerROIConfig(default=(525, 260, 533, 358)),
            "player_7": None,
        },
    },
    "4k": {
        "replay_template": str(_VISION_BASE / "4k" / "replay.jpg"),
        "replay_roi": (1000, 1020, 1150, 1150),
        "dealer_template_dir": str(_VISION_BASE / "4k" / "dealer"),
        "player_template_dir": str(_VISION_BASE / "4k" / "players"),
        "dealer_roi": (1070, 560, 1300, 593),
        # TODO: Set the actual cut-card ROI coordinates for 4k footage
        "cut_card_roi": (1864, 740, 1985, 830),
        "player_rois": {
            "player_1": PlayerROIConfig(
                default=(1045, 1183, 1077, 1467),
                split1=(1003, 1150, 1033, 1445),
                split2=(1090, 1200, 1120, 1490),
            ),
            # TODO: Please fulfill split ROIs for player 2 when you get a chance, I don't have good footage of splits in 4k yet
            "player_2": PlayerROIConfig(default=(1225, 1260, 1255, 1534)),
            "player_3": PlayerROIConfig(
                default=(1440, 1300, 1475, 1580),
                split1=(1400, 1330, 1430, 1575),
                split2=(1484, 1320, 1515, 1588),
            ),
            # TODO: Please fulfill split ROIs for player 4 when you get a chance, I don't have good footage of splits in 4k yet
            "player_4": PlayerROIConfig(default=(1675, 1240, 1705, 1594)),
            "player_5": PlayerROIConfig(
                default=(1910, 1300, 1938, 1578),
                split1=(1867, 1320, 1898, 1589),
                split2=(1953, 1300, 1985, 1578),
            ),
            "player_6": PlayerROIConfig(
                default=(2126, 1270, 2156, 1536),
                split1=(2082, 1290, 2114, 1550),
                split2=(2170, 1280, 2201, 1520),
            ),
            "player_7": PlayerROIConfig(
                default=(2305, 1200, 2335, 1464),
                split1=(2262, 1220, 2295, 1487),
                split2=(2350, 1180, 2380, 1444),
            ),
        },
    },
}

ProfileName = Literal["480p", "4k"]


def build_vision_config(
    video_path: str,
    profile: ProfileName,
    replay_threshold: float = 0.8,
    card_threshold: float = 0.8,
    lookback: int = 5,
    cut_card_threshold: float = 0.15,
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
        lookback=lookback,
        cut_card_roi=defaults["cut_card_roi"],
        cut_card_threshold=cut_card_threshold,
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


def detect_cut_card(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float = 0.15,
) -> bool:
    """Return True when the white cut card is visible inside *roi*.

    Uses a connected-component (blob) approach rather than aggregate pixel
    ratios, exploiting the structural difference between the white cut card
    and caro-pattern card backs:

    - **White cut card**: one large, compact, near-uniform white region.
    - **Caro card back**: many small white blobs scattered between the
      blue/red diamonds — no single blob is large.

    Algorithm:
    1. Convert to HSV and build a binary mask: pixels bright (V > 200)
       *and* near-achromatic (S < 80).  This captures white/cream-tinted
       pixels while excluding vivid skin, diamonds, and table surfaces.
    2. Lightly dilate the mask to bridge small gaps caused by the card's
       tilted edge or slight motion blur.
    3. Find connected components.  If the *largest* single blob covers at
       least *threshold* fraction of the total ROI area, the cut card is
       present.

    *threshold* (default 0.15) means the biggest white blob must occupy
    ≥ 15 % of the ROI.  A caro card's largest white fragment is typically
    < 5 %; the cut card, even partially obscured by the dealer's hand,
    consistently exceeds 15 %.

    Additionally, the largest blob must **not** be left-edge-connected without
    also being right-edge-connected.  The cut card sits inside the shoe on the
    *right* side of the ROI, so its blob always reaches the right border.  The
    dealer's shirt/sleeve enters the ROI from the dealer's body on the *left*,
    producing a blob that touches only the left edge.  Discarding those blobs
    eliminates that class of false positives without affecting true detections.
    """
    x1, y1, x2, y2 = roi
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.uint8((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 80)) * 255

    # Small dilation bridges sub-pixel gaps on the card's tilted edge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)  # type: ignore

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return False

    # stats row 0 is the background; ignore it
    foreground_stats = stats[1:]
    largest_label = int(foreground_stats[:, cv2.CC_STAT_AREA].argmax()) + 1
    largest_blob_area = int(foreground_stats[largest_label - 1, cv2.CC_STAT_AREA])
    total_pixels = region.shape[0] * region.shape[1]

    if largest_blob_area / total_pixels < threshold:
        return False

    # Reject if the largest blob is left-border-connected but NOT right-border-
    # connected.  The cut card sits inside the shoe on the right side of the
    # ROI — its blob always reaches the right edge.  The dealer's shirt/sleeve
    # enters from the dealer's body on the left, producing a blob that touches
    # the left edge only.
    blob_mask = labels == largest_label
    if blob_mask[:, 0].any() and not blob_mask[:, -1].any():
        return False

    return True


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

    Dealer detection runs first — if no dealer cards are found the frame is
    treated as a false-positive replay trigger and ``None`` is returned early.
    Once dealer cards are confirmed, all 7 player ROIs are checked concurrently
    via a thread pool (cv2.matchTemplate releases the GIL).
    """
    session_result: dict = {"frame": frame_idx}

    # Dealer runs first — gates player detection to avoid false positives
    if config.dealer_roi is not None:
        dealer_cards = detect_cards_in_roi(
            frame, config.dealer_roi, dealer_templates, config.card_threshold
        )
        session_result["dealer"] = dealer_cards
        if not dealer_cards:
            return None
        logger.debug("Dealer cards: %s", dealer_cards)
    else:
        session_result["dealer"] = []
        logger.debug("Dealer ROI not configured, skipping")

    # Player detections run concurrently now that dealer is confirmed
    player_tasks: dict[str, PlayerROIConfig] = {
        f"player_{i}": roi_cfg
        for i in range(1, 8)
        if (roi_cfg := config.player_rois.get(f"player_{i}")) is not None
    }

    def _detect_player(roi_cfg: PlayerROIConfig) -> dict[str, list[str]]:
        """Return a hands dict for one player using split-aware short-circuit logic.

        - If ``default`` ROI has cards → return ``{"hand1": cards}``.
        - Otherwise try ``split1``; if cards found, also try ``split2``.
        - Returns ``{}`` when no cards are detected in any ROI.
        """
        if roi_cfg.default is not None:
            cards = detect_cards_in_roi(
                frame, roi_cfg.default, player_templates, config.card_threshold
            )
            if cards:
                return {"hand1": cards}
        if roi_cfg.split1 is not None:
            split1_cards = detect_cards_in_roi(
                frame, roi_cfg.split1, player_templates, config.card_threshold
            )
            if split1_cards:
                hands: dict[str, list[str]] = {"hand1": split1_cards}
                if roi_cfg.split2 is not None:
                    split2_cards = detect_cards_in_roi(
                        frame, roi_cfg.split2, player_templates, config.card_threshold
                    )
                    if split2_cards:
                        hands["hand2"] = split2_cards
                return hands
        return {}

    with ThreadPoolExecutor(max_workers=len(player_tasks) or 1) as executor:
        futures = {
            key: executor.submit(_detect_player, roi_cfg)
            for key, roi_cfg in player_tasks.items()
        }
        player_results = {key: fut.result() for key, fut in futures.items()}

    for i in range(1, 8):
        key = f"player_{i}"
        session_result[key] = player_results.get(key, {})
        if session_result[key]:
            logger.debug("Player %d cards: %s", i, session_result[key])

    return session_result


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

_DEBUG_DIR = Path(__file__).parent.parent.parent / "data" / "debug_frames"


def _save_debug_frame(frame: np.ndarray, frame_index: int, label: str) -> None:
    """Save *frame* to ``data/debug_frames/<label>_<frame_index>.jpg``.

    Failures are logged at WARNING level and never propagate to the caller.
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        filename = _DEBUG_DIR / f"{label}_{frame_index:08d}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.debug("Debug frame saved: %s", filename)
    except Exception as exc:
        logger.warning("Could not save debug frame: %s", exc)


# ---------------------------------------------------------------------------
# Main video processing
# ---------------------------------------------------------------------------


class LiveVideoProcessor:
    """Stateful processor for continuous frame-by-frame analysis."""

    def __init__(
        self,
        config: VisionConfig,
        dealer_templates: list[tuple[str, np.ndarray]],
        player_templates: list[tuple[str, np.ndarray]],
    ):
        self.config = config
        self.dealer_templates = dealer_templates
        self.player_templates = player_templates

        self.replay_template_img = cv2.imread(
            self.config.replay_template, cv2.IMREAD_GRAYSCALE
        )
        if self.replay_template_img is None:
            raise FileNotFoundError(
                f"Could not read replay template: {self.config.replay_template}"
            )

        self.LOOKBACK = self.config.lookback
        self.buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=self.LOOKBACK + 1)

        self.frame_index = 0
        self.session_num = 1
        self.replay_active = False

        # Cut-card / deck-swap state
        # pending_deck_swap is set when the white cut card is seen and cleared
        # after the current game session's result is emitted.  deck_num starts
        # at 1 and increments at each session boundary where a swap was detected.
        self.pending_deck_swap: bool = False
        self.deck_num: int = 1

        # Consecutive-swap cancellation: if two back-to-back sessions both
        # trigger a deck swap the first was a false positive.  We save the
        # pre-swap state here and restore it if the very next session also
        # fires a swap, so the emitted result is stamped with the correct
        # deck/session numbers before the real swap is applied.
        self._last_session_swapped: bool = False
        self._pre_swap_deck_num: int = 1
        self._pre_swap_session_num: int = 1

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """Process a single incoming frame and return a session dict if detected."""
        self.buffer.append((self.frame_index, frame.copy()))

        # ------------------------------------------------------------------
        # Cut-card detection — runs every frame while not already pending.
        # Once the cut card is spotted we stop looking until the current
        # game session ends so we don't double-count.
        # ------------------------------------------------------------------
        if (
            self.config.cut_card_roi is not None
            and not self.pending_deck_swap
            and detect_cut_card(
                frame,
                self.config.cut_card_roi,
                self.config.cut_card_threshold,
            )
        ):
            self.pending_deck_swap = True
            logger.warning(
                "Frame %d: white cut card detected — deck swap pending after session %d",
                self.frame_index,
                self.session_num,
            )
            _save_debug_frame(frame, self.frame_index, "cut_card")

        if self.config.replay_roi is not None:
            x1, y1, x2, y2 = self.config.replay_roi
            search_region = frame[y1:y2, x1:x2]
        else:
            search_region = frame

        gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        match_result = cv2.matchTemplate(
            gray_region,
            self.replay_template_img,  # type: ignore
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, _ = cv2.minMaxLoc(match_result)

        session_result_out = None

        if max_val >= self.config.replay_threshold and not self.replay_active:
            self.replay_active = True
            target_idx, target_frame = self.buffer[0]

            session_result = _detect_cards_for_session(
                target_idx,
                target_frame,
                self.config,
                self.dealer_templates,
                self.player_templates,
            )

            if session_result is not None:
                # Consecutive-swap cancellation: if the previous session also
                # triggered a deck swap AND this session triggers one too, the
                # previous swap was a false positive.  Restore the pre-swap
                # deck/session state before stamping this result so the emitted
                # dict carries the correct numbers.
                if self.pending_deck_swap and self._last_session_swapped:
                    logger.warning(
                        "Frame %d: two consecutive deck-swap triggers detected — "
                        "cancelling previous false-positive swap (restoring deck %d, session %d)",
                        self.frame_index,
                        self._pre_swap_deck_num,
                        self._pre_swap_session_num,
                    )
                    self.deck_num = self._pre_swap_deck_num
                    self.session_num = self._pre_swap_session_num
                    self._last_session_swapped = False

                logger.info(
                    "Session %d - Deck %d: replay at frame %d, cards detected at frame %d",
                    self.session_num,
                    self.deck_num,
                    self.frame_index,
                    target_idx,
                )
                session_result_out = {
                    "session": self.session_num,
                    "deck_num": self.deck_num,
                    **session_result,
                }
                self.session_num += 1

                # Deck swap: current session was played with deck_num; the
                # next session will use the new deck.  Reset the flag so
                # cut-card detection re-arms for the following session.
                if self.pending_deck_swap:
                    # Save undo state before modifying deck/session counters.
                    self._pre_swap_deck_num = self.deck_num
                    self._pre_swap_session_num = self.session_num
                    self.deck_num += 1
                    self.pending_deck_swap = False
                    self._last_session_swapped = True
                    logger.info(
                        "Deck swap confirmed after session %d — now on deck %d",
                        self.session_num - 1,
                        self.deck_num,
                    )

                    # Reset the session counter to 1 when the deck number increments, so the session numbers are per-deck and easier to correlate with physical shoe changes in the footage.
                    self.session_num = 1
                else:
                    self._last_session_swapped = False

        elif max_val < self.config.replay_threshold and self.replay_active:
            self.replay_active = False

        self.frame_index += 1
        return session_result_out


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
    processor = LiveVideoProcessor(config, dealer_templates, player_templates)

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
        "Processing video: %s  [%dx%d]  replay_threshold=%.2f  card_threshold=%.2f  cut_card_threshold=%.2f",
        config.video_path,
        frame_width,
        frame_height,
        config.replay_threshold,
        config.card_threshold,
        config.cut_card_threshold,
    )

    results: list[dict] = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = processor.process_frame(frame)
        if result is not None:
            results.append(result)

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


def map_player_hands(
    hands: dict[str, list[str]],
) -> dict[str, list[CardClass]] | None:
    """Map a player hands dict of raw card name stems to ``CardClass`` values.

    Returns ``None`` when *hands* is empty (no cards detected for that player).
    """
    if not hands:
        return None
    return {hand: map_card_names(cards) for hand, cards in hands.items()}
