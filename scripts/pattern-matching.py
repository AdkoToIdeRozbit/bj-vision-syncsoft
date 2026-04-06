import argparse
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision.processor import (
    LiveVideoProcessor,
    ProfileName,
    VisionConfig,
    build_vision_config,
    load_card_templates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> tuple[VisionConfig, dict]:
    """Load configuration from .env file with optional CLI overrides.

    Returns a ``(VisionConfig, script_config)`` tuple where *script_config*
    holds output-related settings not part of ``VisionConfig``.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Detect game sessions and identify cards via template matching.",
    )
    parser.add_argument(
        "--video", default=None, help="Path to input video (overrides .env VIDEO_PATH)"
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["480p", "4k"],
        help="Resolution profile (overrides .env PROFILE)",
    )
    parser.add_argument(
        "--replay-threshold",
        type=float,
        default=None,
        help="Replay detection threshold (overrides .env REPLAY_THRESHOLD)",
    )
    parser.add_argument(
        "--card-threshold",
        type=float,
        default=None,
        help="Card detection threshold (overrides .env CARD_THRESHOLD)",
    )
    parser.add_argument(
        "--output-prefix",
        default="session",
        help="Output file name prefix",
    )
    parser.add_argument(
        "--output-dir",
        default="game-sessions",
        help="Output directory for session frames and results.json ",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        help="Image file extension for saved frames",
    )
    args = parser.parse_args()

    video_path = args.video or os.getenv("VIDEO_PATH")
    if not video_path:
        raise ValueError("Video path required: set VIDEO_PATH in .env or pass --video")

    profile: ProfileName = args.profile or os.getenv("PROFILE", "4k")  # type: ignore[assignment]
    replay_threshold = args.replay_threshold or float(
        os.getenv("REPLAY_THRESHOLD", "0.8")
    )
    card_threshold = args.card_threshold or float(os.getenv("CARD_THRESHOLD", "0.8"))

    vision_config = build_vision_config(
        video_path=video_path,
        profile=profile,
        replay_threshold=replay_threshold,
        card_threshold=card_threshold,
    )

    script_config = {
        "output_dir": args.output_dir,
        "output_prefix": args.output_prefix,
        "image_ext": args.image_ext,
    }

    return vision_config, script_config


# ---------------------------------------------------------------------------
# Session detection + inline card detection
# ---------------------------------------------------------------------------


def process_video_with_frames(
    config: VisionConfig,
    dealer_templates: list[tuple[str, np.ndarray]],
    player_templates: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    image_ext: str,
) -> list[dict]:
    """Scan video for replay-button appearances, detect cards inline, and save frames.

    Wraps ``LiveVideoProcessor`` to add session-ending frame persistence on top
    of the core detection logic. Returns a list of session result dicts.
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
        "Processing video: %s  [%dx%d]  replay_threshold=%.2f  card_threshold=%.2f",
        config.video_path,
        frame_width,
        frame_height,
        config.replay_threshold,
        config.card_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Maintain a local frame buffer that mirrors the processor's internal buffer
    # so we can retrieve the lookback frame for saving when a session is detected.
    buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=config.lookback + 1)
    local_frame_index = 0
    results: list[dict] = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        buffer.append((local_frame_index, frame))
        result = processor.process_frame(frame)

        if result is not None:
            _, target_frame = buffer[0]
            session_num = result["session"]
            frame_idx = result["frame"]
            file_name = f"{prefix}_{session_num}_frame_{frame_idx:06d}.{image_ext}"
            file_path = output_dir / file_name
            cv2.imwrite(str(file_path), target_frame)
            logger.info(
                "Session %d: replay detected at frame %d, saved frame %d → %s",
                session_num,
                local_frame_index,
                frame_idx,
                file_path,
            )
            results.append(result)

        local_frame_index += 1

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    vision_config, script_config = load_config()

    dealer_templates = load_card_templates(vision_config.dealer_template_dir)
    player_templates = load_card_templates(vision_config.player_template_dir)
    logger.info(
        "Loaded %d dealer template(s), %d player template(s)",
        len(dealer_templates),
        len(player_templates),
    )

    out_dir = Path(script_config["output_dir"])
    results = process_video_with_frames(
        vision_config,
        dealer_templates,
        player_templates,
        output_dir=out_dir,
        prefix=script_config["output_prefix"],
        image_ext=script_config["image_ext"],
    )

    if not results:
        logger.warning(
            "No replay button detected — no game session endings found.\n"
            "Possible causes:\n"
            "  - The video ends before the replay button appears\n"
            "  - REPLAY_THRESHOLD is too high\n"
            "  - The selected profile's REPLAY_ROI does not cover the replay button area"
        )
        return

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("%d session(s) processed.", len(results))
    logger.info("Results written to %s", json_path)
    logger.info("Session frames saved to %s/", out_dir)


if __name__ == "__main__":
    main()
