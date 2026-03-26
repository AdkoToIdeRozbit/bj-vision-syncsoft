"""
This script extracts frames from a video file and saves them as individual image files in an output directory.
You can specify how frequently to save frames (e.g., every N frames) and customize the output filename format and image extension.

Example usage: python extract_frames.py --video input.mp4 --output-dir frames --every-n-frames 10 --prefix frame --image-ext png

This will save every 10th frame from input.mp4 as a PNG image in the "frames" directory, with filenames like frame_000000.png, frame_000010.png, etc.
"""

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file into an output directory."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--output-dir",
        default="extracted-frames",
        help="Directory where extracted frames will be saved",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="Save one frame every N frames (default: 1 = save all)",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "png", "jpeg", "webp"],
        help="Output image extension",
    )
    return parser.parse_args()


def extract_frames(
    video_path: str,
    output_dir: str,
    every_n_frames: int,
    prefix: str,
    image_ext: str,
) -> tuple[int, int]:
    if every_n_frames <= 0:
        raise ValueError("--every-n-frames must be greater than 0")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_index = 0
    saved_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % every_n_frames == 0:
            # cropped_frame = frame[320:1080, 350:1500]
            cropped_frame = frame

            file_name = f"{prefix}_{saved_count:06d}.{image_ext}"
            file_path = out_dir / file_name
            cv2.imwrite(str(file_path), cropped_frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return frame_index, saved_count


def main() -> None:
    args = parse_args()
    total_frames, saved_frames = extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        every_n_frames=args.every_n_frames,
        prefix=args.prefix,
        image_ext=args.image_ext,
    )
    print(f"Processed frames: {total_frames}")
    print(f"Saved frames: {saved_frames}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
