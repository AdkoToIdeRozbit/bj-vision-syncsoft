"""
Diagnostic script for the cut-card detector.

Analyses both the true-positive and false-positive debug frames and prints
a per-frame breakdown of the HSV statistics used by detect_cut_card(), so we
can see exactly why each frame triggers or doesn't.

Usage (from repo root, venv activated):
    python scripts/diagnose_cut_card.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Frames and ROI (4k profile)
# ---------------------------------------------------------------------------

FRAMES = {
    "true_positive  (frame  936)": "data/debug_frames/cut_card_00000936.jpg",
    "true_positive  (frame 1807)": "data/debug_frames/cut_card_00001807.jpg",
    "false_positive (frame 5010)": "data/debug_frames/cut_card_00005010.jpg",
    "false_positive (frame 10023)": "data/debug_frames/cut_card_00010023.jpg",
}

ROI = (1864, 740, 1985, 830)  # 4k cut_card_roi: (x1, y1, x2, y2)
THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def analyse(
    label: str, image_path: str, roi: tuple[int, int, int, int], threshold: float
) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read {image_path}")
        return

    x1, y1, x2, y2 = roi
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        print(
            f"\n[WARN] ROI {roi} is out of bounds for image {frame.shape[:2]} — skipping {label}"
        )
        return
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    total = region.shape[0] * region.shape[1]

    # ------------------------------------------------------------------
    # Old (original) algorithm — V > 200 only
    # ------------------------------------------------------------------
    old_bright = np.count_nonzero(hsv[:, :, 2] > 200)
    old_ratio = old_bright / total
    old_result = old_ratio >= threshold

    # ------------------------------------------------------------------
    # New algorithm — largest white blob
    # ------------------------------------------------------------------
    mask = np.uint8((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 80)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)  # type: ignore
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_dilated, connectivity=8
    )
    if n_labels >= 2:
        blob_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_blob = int(blob_areas.max())
        second_largest = int(sorted(blob_areas)[-2]) if len(blob_areas) > 1 else 0
        num_blobs = len(blob_areas)
    else:
        largest_blob = second_largest = num_blobs = 0
    blob_ratio = largest_blob / total
    new_result = blob_ratio >= threshold

    # ------------------------------------------------------------------
    # Border-touch check (fix for dealer-shirt false positives)
    # ------------------------------------------------------------------
    if n_labels >= 2:
        largest_label = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
        # re-run with label image
        _, labels_img, stats2, _ = cv2.connectedComponentsWithStats(
            mask_dilated, connectivity=8
        )
        blob_mask = labels_img == largest_label
        touches_border = (
            blob_mask[0, :].any()
            or blob_mask[-1, :].any()
            or blob_mask[:, 0].any()
            or blob_mask[:, -1].any()
        )
        # Which specific borders does it touch?
        touches_top = bool(blob_mask[0, :].any())
        touches_bottom = bool(blob_mask[-1, :].any())
        touches_left = bool(blob_mask[:, 0].any())
        touches_right = bool(blob_mask[:, -1].any())
        # Bounding box of the largest blob
        bb_left = int(stats2[largest_label, cv2.CC_STAT_LEFT])
        bb_top = int(stats2[largest_label, cv2.CC_STAT_TOP])
        bb_w = int(stats2[largest_label, cv2.CC_STAT_WIDTH])
        bb_h = int(stats2[largest_label, cv2.CC_STAT_HEIGHT])
        roi_h, roi_w = region.shape[:2]
        fill_ratio = largest_blob / (bb_w * bb_h) if bb_w * bb_h > 0 else 0
    else:
        touches_border = touches_top = touches_bottom = touches_left = touches_right = (
            False
        )
        bb_left = bb_top = bb_w = bb_h = roi_w = roi_h = 0
        fill_ratio = 0
    fixed_result = new_result and not (touches_left and not touches_right)

    # ------------------------------------------------------------------
    # Saturation histogram buckets (useful for tuning)
    # ------------------------------------------------------------------
    s_channel = hsv[:, :, 1].flatten()
    buckets = [(0, 40), (40, 80), (80, 120), (120, 200), (200, 256)]
    s_hist = {
        f"S {lo}-{hi}": int(np.count_nonzero((s_channel >= lo) & (s_channel < hi)))
        for lo, hi in buckets
    }

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  File : {image_path}")
    print(f"  ROI  : {roi}  →  {region.shape[1]}×{region.shape[0]} px  ({total} total)")
    print(f"{'=' * 60}")
    print(
        f"  V-channel stats  min={hsv[:, :, 2].min():3d}  max={hsv[:, :, 2].max():3d}  "
        f"mean={hsv[:, :, 2].mean():.1f}"
    )
    print(
        f"  S-channel stats  min={hsv[:, :, 1].min():3d}  max={hsv[:, :, 1].max():3d}  "
        f"mean={hsv[:, :, 1].mean():.1f}"
    )
    print()
    print("  Saturation histogram:")
    for bucket, count in s_hist.items():
        bar = "█" * int(count / total * 40)
        print(
            f"    {bucket:>12s} : {count:5d} px  ({count / total * 100:5.1f}%)  {bar}"
        )
    print()
    print("  OLD algorithm  (V>200 only)")
    print(
        f"    bright pixels : {old_bright:5d}  ({old_ratio * 100:.1f}%)  threshold={threshold}"
    )
    print(f"    result        : {'✓ DETECTED' if old_result else '✗ not detected'}")
    print()
    print("  NEW algorithm  (largest white blob)")
    print(f"    white blobs   : {num_blobs}")
    print(f"    largest blob  : {largest_blob:5d} px  ({blob_ratio * 100:.1f}%)")
    print(
        f"    2nd largest   : {second_largest:5d} px  ({second_largest / total * 100:.1f}%)"
    )
    print(f"    threshold     : {threshold}  ({threshold * 100:.0f}% of ROI)")
    print(f"    blob>=thr     : {new_result}  ({blob_ratio:.3f} >= {threshold})")
    print(f"    result        : {'✓ DETECTED' if new_result else '✗ not detected'}")
    print()
    print("  FIXED algorithm  (+ border-touch rejection)")
    print(
        f"    touches border: {touches_border}  (T={touches_top} B={touches_bottom} L={touches_left} R={touches_right})"
    )
    print(
        f"    blob bbox     : x={bb_left} y={bb_top} w={bb_w} h={bb_h}  (ROI {roi_w}×{roi_h})"
    )
    print(f"    bbox fill     : {fill_ratio:.2f}  (blob area / bbox area)")
    print(f"    result        : {'✓ DETECTED' if fixed_result else '✗ not detected'}")

    # Save annotated ROI crop + white mask for visual inspection
    out_dir = Path("data/debug_frames")
    tag = (
        label.split("(")[1].rstrip(")").replace(" ", "_")
        if "(" in label
        else label.replace(" ", "_")
    )
    roi_bgr = region.copy()
    cv2.rectangle(
        roi_bgr, (0, 0), (roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1), (0, 255, 0), 1
    )
    cv2.imwrite(str(out_dir / f"diagnose_roi_{tag}.png"), roi_bgr)
    cv2.imwrite(str(out_dir / f"diagnose_mask_{tag}.png"), mask_dilated)
    print(f"\n  ROI crop saved → data/debug_frames/diagnose_roi_{tag}.png")
    print(f"  Mask saved     → data/debug_frames/diagnose_mask_{tag}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    for label, rel_path in FRAMES.items():
        analyse(label, str(repo_root / rel_path), ROI, THRESHOLD)
    print()
