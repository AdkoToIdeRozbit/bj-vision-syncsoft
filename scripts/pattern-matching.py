import argparse
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


def parse_roi(roi_str: str) -> tuple[int, int, int, int]:
    """Parse ROI string 'x1,y1,x2,y2' into a tuple of ints."""
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"ROI must have exactly 4 comma-separated values, got: {roi_str}"
        )
    x1, y1, x2, y2 = (int(p.strip()) for p in parts)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"Invalid ROI: x1 must be < x2 and y1 must be < y2, got ({x1},{y1},{x2},{y2})"
        )
    return x1, y1, x2, y2


def load_config() -> dict:
    """Load configuration from .env file with optional --video CLI override."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Detect game sessions and identify cards via template matching.",
    )
    parser.add_argument(
        "--video", default=None, help="Path to input video (overrides .env VIDEO_PATH)"
    )
    args = parser.parse_args()

    video_path = args.video or os.getenv("VIDEO_PATH")
    if not video_path:
        raise ValueError("Video path required: set VIDEO_PATH in .env or pass --video")

    replay_roi_str = os.getenv("REPLAY_ROI")
    dealer_roi_str = os.getenv("DEALER_ROI")

    player_rois: dict[str, tuple[int, int, int, int] | None] = {}
    for i in range(1, 8):
        roi_str = os.getenv(f"PLAYER_{i}_ROI")
        player_rois[f"player_{i}"] = parse_roi(roi_str) if roi_str else None

    return {
        "video_path": video_path,
        "replay_template": os.getenv("REPLAY_TEMPLATE", "template-images/replay.jpg"),
        "replay_roi": parse_roi(replay_roi_str) if replay_roi_str else None,
        "replay_threshold": float(os.getenv("REPLAY_THRESHOLD", "0.8")),
        "card_threshold": float(os.getenv("CARD_THRESHOLD", "0.8")),
        "dealer_template_dir": os.getenv(
            "DEALER_TEMPLATE_DIR", "template-images/dealer"
        ),
        "player_template_dir": os.getenv(
            "PLAYER_TEMPLATE_DIR", "template-images/players"
        ),
        "dealer_roi": parse_roi(dealer_roi_str) if dealer_roi_str else None,
        "player_rois": player_rois,
        "output_dir": os.getenv("OUTPUT_DIR", "game-sessions"),
        "output_prefix": os.getenv("OUTPUT_PREFIX", "session"),
        "image_ext": os.getenv("IMAGE_EXT", "jpg"),
    }


# ---------------------------------------------------------------------------
# Session detection + inline card detection
# ---------------------------------------------------------------------------


def process_video(
    config: dict,
    dealer_templates: list[tuple[str, np.ndarray]],
    player_templates: list[tuple[str, np.ndarray]],
) -> list[dict]:
    """Scan video for replay-button appearances and detect cards immediately.

    For each session-ending frame detected, card detection runs inline so
    results are produced as the video is processed (no second pass needed).

    Returns a list of session result dicts.
    """
    video_path = config["video_path"]
    template_path = config["replay_template"]
    replay_roi = config["replay_roi"]
    replay_threshold = config["replay_threshold"]
    card_threshold = config["card_threshold"]
    out_dir = Path(config["output_dir"])
    prefix = config["output_prefix"]
    image_ext = config["image_ext"]

    replay_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if replay_template is None:
        raise FileNotFoundError(f"Could not read template image: {template_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if replay_roi is not None:
        x1, y1, x2, y2 = replay_roi
        if x2 > frame_width or y2 > frame_height:
            raise ValueError(
                f"Replay ROI ({x1},{y1},{x2},{y2}) exceeds frame dimensions "
                f"({frame_width}x{frame_height})"
            )

    print(f"Processing video: {video_path}")
    # print(f"Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}")
    print(f"Replay template: {template_path}, Threshold: {replay_threshold}")
    if replay_roi is not None:
        print(
            f"Replay ROI: ({replay_roi[0]},{replay_roi[1]}) to ({replay_roi[2]},{replay_roi[3]})"
        )
    print(f"Card detection threshold: {card_threshold}")

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

        if replay_roi is not None:
            x1, y1, x2, y2 = replay_roi
            search_region = frame[y1:y2, x1:x2]
        else:
            search_region = frame

        gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_region, replay_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val >= replay_threshold and not replay_active:
            replay_active = True
            target_idx, target_frame = buffer[0]

            # Detect cards inline
            session_result = _detect_cards_for_session(
                target_idx,
                target_frame,
                config,
                dealer_templates,
                player_templates,
                card_threshold,
            )

            # Only save session frame and results if card detection was successful (dealer cards detected)
            if session_result is not None:
                # Save session-ending frame image
                file_name = f"{prefix}_{session_num}_frame_{target_idx:06d}.{image_ext}"
                file_path = out_dir / file_name
                cv2.imwrite(str(file_path), target_frame)
                print(
                    f"Session {session_num}: replay detected at frame {frame_index}, "
                    f"saved frame {target_idx} → {file_path}\n"
                )

                results.append({"session": session_num, **session_result})
                session_num = len(results) + 1

        elif max_val < replay_threshold and replay_active:
            replay_active = False

        frame_index += 1

    cap.release()
    return results


def _detect_cards_for_session(
    frame_idx: int,
    frame: np.ndarray,
    config: dict,
    dealer_templates: list[tuple[str, np.ndarray]],
    player_templates: list[tuple[str, np.ndarray]],
    card_threshold: float,
) -> dict | None:
    """Detect dealer and player cards in a single session-ending frame."""
    session_result: dict = {
        "frame": frame_idx,
    }

    # Dealer cards
    dealer_roi = config["dealer_roi"]
    if dealer_roi is not None:
        dealer_cards = detect_cards_in_roi(
            frame, dealer_roi, dealer_templates, card_threshold
        )
        session_result["dealer"] = dealer_cards

        # If no dealer cards detected, likely a false replay detection → skip this session
        if not dealer_cards:
            return None

        print(f"    Dealer: {dealer_cards}")
    else:
        session_result["dealer"] = []
        print("    Dealer: ROI not configured, skipping")

    # Player cards (1–7)
    for i in range(1, 8):
        key = f"player_{i}"
        player_roi = config["player_rois"].get(key)
        if player_roi is not None:
            player_cards = detect_cards_in_roi(
                frame, player_roi, player_templates, card_threshold
            )
            session_result[key] = player_cards
            print(f"    Player {i}: {player_cards}")
        else:
            session_result[key] = []

    return session_result


# ---------------------------------------------------------------------------
# Phase 2: Card detection in session-ending frames
# ---------------------------------------------------------------------------


def load_card_templates(directory: str) -> list[tuple[str, np.ndarray]]:
    """Load all image files from a directory as grayscale templates.

    Returns a list of (card_name, grayscale_image) tuples.
    Card name is the filename without extension (e.g., 'king-red.png' → 'king-red').
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
        print(f"  Warning: no templates found in {directory}")
    return templates


# def load_card_templates(directory: str) -> list[tuple[str, np.ndarray]]:
#     """Load all image files from a directory as color (BGR) templates.

#     Returns a list of (card_name, color_image) tuples.
#     Card name is the filename without extension (e.g., 'king-red.png' → 'king-red').
#     """
#     templates: list[tuple[str, np.ndarray]] = []
#     dir_path = Path(directory)
#     if not dir_path.is_dir():
#         raise FileNotFoundError(f"Template directory not found: {directory}")

#     for file in sorted(dir_path.iterdir()):
#         if file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
#             img = cv2.imread(str(file), cv2.IMREAD_COLOR)
#             if img is not None:
#                 templates.append((file.stem, img))

#     if not templates:
#         print(f"  Warning: no templates found in {directory}")
#     return templates


def _merge_detections(
    points: list[tuple[int, int]], template_w: int, template_h: int
) -> int:
    """Merge overlapping detection points into distinct clusters.

    Points within one template-width/height distance belong to the same physical card.
    Returns the number of distinct card instances.
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

    Groups overlapping detections of the same template into distinct card instances.
    Returns a list of detected card names (may contain duplicates for multiple physical cards).
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

        # Group overlapping detections into distinct physical cards
        points = list(zip(locations[1].tolist(), locations[0].tolist()))
        merged_count = _merge_detections(points, tw, th)
        for _ in range(merged_count):
            detected_cards.append(card_name)

    return detected_cards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()

    # Load all card templates into RAM upfront
    dealer_templates = load_card_templates(config["dealer_template_dir"])
    player_templates = load_card_templates(config["player_template_dir"])
    print(
        f"Loaded {len(dealer_templates)} dealer template(s), "
        f"{len(player_templates)} player template(s)"
    )

    # Process video: detect sessions and cards in a single pass
    results = process_video(config, dealer_templates, player_templates)

    if not results:
        print("\nNo replay button detected — no game session endings found.")
        print("Possible causes:")
        print("  - The video ends before the replay button appears")
        print("  - REPLAY_THRESHOLD is too high")
        print("  - REPLAY_ROI does not cover the replay button area")
        return

    # Write JSON output
    out_dir = Path(config["output_dir"])
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} session(s) processed.")
    print(f"Results written to {json_path}")
    print(f"Session frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
