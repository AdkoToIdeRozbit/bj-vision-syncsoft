"""
This script detects game session endings in a video by template-matching a replay button, then outputs the 5th-previous frame (last frame of each game session).
Default ROI for replay button: 1000,1020,1150,1150 (for 3248 x 2122 videos) (tuned for 1920x1080 videos, adjust as needed)

Example usage: python extract-session-ending-frames.py --video input.mp4 --template template-images/4k/replay.jpg --roi 1000,1020,1150,1150 --threshold 0.8 --output-dir game-sessions --prefix session --image-ext jpg
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect game session endings in a video by template-matching a replay button, "
            "then output the 5th-previous frame (last frame of each game session)."
        ),
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--template",
        default="template-images/4k/replay.jpg",
        help="Path to replay-button template image (default: template-images/4k/replay.jpg)",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Region of interest for template matching, format: x1,y1,x2,y2. "
            "Limits the search area to reduce computation. "
            "Coordinates are top-left (x1,y1) and bottom-right (x2,y2)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Template matching confidence threshold (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        default="game-sessions",
        help="Directory where session-ending frames will be saved (default: game-sessions)",
    )
    parser.add_argument(
        "--prefix",
        default="session",
        help="Output filename prefix (default: session)",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "png", "jpeg", "webp"],
        help="Output image extension (default: jpg)",
    )
    return parser.parse_args()


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


def detect_session_endings(
    video_path: str,
    template_path: str,
    roi: tuple[int, int, int, int] | None,
    threshold: float,
    output_dir: str,
    prefix: str,
    image_ext: str,
) -> int:
    """Detect replay-button appearances and save the 5th-previous frame for each.

    Returns the number of game sessions detected.
    """
    # Load template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Could not read template image: {template_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rolling buffer: stores (frame_index, frame) tuples.
    # We keep up to 6 entries so that when the reload button is detected on
    # frame N, the 5th-previous frame (N-5) is at buffer[0] when the buffer
    # is full (indices: N-5, N-4, N-3, N-2, N-1, N).
    LOOKBACK = 5
    buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=LOOKBACK + 1)

    frame_index = 0
    session_count = 0
    replay_active = False  # debounce flag

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Validate ROI against frame dimensions
    if roi is not None:
        x1, y1, x2, y2 = roi
        if x2 > frame_width or y2 > frame_height:
            raise ValueError(
                f"ROI ({x1},{y1},{x2},{y2}) exceeds frame dimensions ({frame_width}x{frame_height})"
            )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}")
    print(f"Template: {template_path}, Threshold: {threshold}")
    if roi is not None:
        print(f"ROI: ({roi[0]},{roi[1]}) to ({roi[2]},{roi[3]})")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Store (frame_index, full frame) in the rolling buffer
        buffer.append((frame_index, frame))

        # Extract the region to match against
        if roi is not None:
            x1, y1, x2, y2 = roi
            search_region = frame[y1:y2, x1:x2]
        else:
            search_region = frame

        # Convert to grayscale for template matching
        gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val >= threshold and not replay_active:
            # First frame where replay button appears → end of a game session
            replay_active = True
            session_count += 1

            # Grab the 5th-previous frame from the buffer.
            # Buffer stores up to LOOKBACK+1 entries. The target frame
            # is at index 0 when the buffer is full, otherwise the earliest
            # available frame.
            target_idx, target_frame = buffer[0]

            file_name = f"{prefix}_{session_count}_frame_{target_idx:06d}.{image_ext}"
            file_path = out_dir / file_name
            cv2.imwrite(str(file_path), target_frame)
            print(
                f"  Session {session_count}: replay detected at frame {frame_index}, "
                f"saved frame {target_idx} → {file_path}"
            )

        elif max_val < threshold and replay_active:
            # Replay button has disappeared → ready for the next session
            replay_active = False

        frame_index += 1

    cap.release()
    return session_count


def main() -> None:
    args = parse_args()

    roi = parse_roi(args.roi) if args.roi is not None else None

    session_count = detect_session_endings(
        video_path=args.video,
        template_path=args.template,
        roi=roi,
        threshold=args.threshold,
        output_dir=args.output_dir,
        prefix=args.prefix,
        image_ext=args.image_ext,
    )

    if session_count == 0:
        print(
            "\nNo reload button detected in the video — no game session endings found."
        )
        print("Possible causes:")
        print("  - The video ends before the reload button appears")
        print("  - The threshold is too high (try lowering --threshold)")
        print("  - The ROI does not cover the reload button area")
    else:
        print(
            f"\nDone. {session_count} game session(s) detected. Frames saved to: {args.output_dir}/"
        )


if __name__ == "__main__":
    main()
