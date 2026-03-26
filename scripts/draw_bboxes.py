"""
This script draws bounding boxes on an image given a list of boxes in xyxy format (absolute pixel coordinates).

Example usage: python draw_bboxes.py input.jpg 100,100,200,200 300,300,400,400 --output output.jpg --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 128),
]


def draw_bboxes(image: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Draw bounding boxes in xyxy format (absolute pixel coordinates)."""
    result = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw bounding boxes on an image")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "boxes",
        nargs="+",
        help="Bounding boxes in xyxy format: x1,y1,x2,y2 (repeatable)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <input>_bbox.<ext>)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the result in a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    xyxy_boxes = []
    for b in args.boxes:
        coords = [float(v) for v in b.split(",")]
        if len(coords) != 4:
            raise ValueError(f"Expected x1,y1,x2,y2 but got: {b}")
        xyxy_boxes.append(tuple(coords))
    result = draw_bboxes(image, xyxy_boxes)

    # Save the result to disk
    # output_path = args.output
    # if not output_path:
    #     p = Path(args.image)
    #     output_path = str(p.with_stem(p.stem + "_bbox"))

    # cv2.imwrite(output_path, result)
    # print(f"Saved to {output_path}")

    if args.show:
        cv2.imshow("Bounding Boxes", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
