"""
Threshold-based Foci Classifier.

Applies a morphological pipeline (NLM Denoising -> Top-Hat -> CLAHE -> Area Opening)
to isolate foci, then counts foci per segmented cell and classifies each cell as
Healthy or Damaged based on a damage threshold.

Usage:
    python classify_foci_threshold.py \
        --image_path  path/to/image.tif \
        --mask_path   path/to/image_mask.png \
        --foci_channel 1 \
        [--denoise_h 15] \
        [--tophat_size 13] \
        [--clip_limit 6.0] \
        [--min_area 11] \
        [--damage_threshold 5] \
        [--save_foci_mask]

Output (stdout): JSON with per-cell classification results.
If --save_foci_mask is set, also saves <image>_foci_mask.png next to the mask.
"""

import argparse
import json
import sys
import os

import numpy as np
from PIL import Image

try:
    import cv2 as cv
except ImportError:
    print(json.dumps({"error": "opencv-python not installed. Run: pip install opencv-python"}))
    sys.exit(1)

try:
    from skimage import morphology, measure
except ImportError:
    print(json.dumps({"error": "scikit-image not installed. Run: pip install scikit-image"}))
    sys.exit(1)


def normalize_to_uint8(channel_data):
    """Normalize any-bit-depth channel data to uint8 [0, 255]."""
    arr = channel_data.astype(np.float64)
    p_low = np.percentile(arr, 0.5)
    p_high = np.percentile(arr, 99.5)
    if p_high - p_low < 1e-6:
        return np.zeros_like(channel_data, dtype=np.uint8)
    arr = (arr - p_low) / (p_high - p_low)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def extract_channel(image_path, foci_channel):
    """Extract a specific channel from a multi-channel image as uint8 grayscale."""
    raw_image = Image.open(image_path)

    if foci_channel < 0:
        # Auto: use first frame
        arr = np.array(raw_image)
        if arr.dtype != np.uint8:
            return normalize_to_uint8(arr)
        return arr if arr.ndim == 2 else cv.cvtColor(arr, cv.COLOR_RGB2GRAY)

    # Multi-frame TIFF
    n_frames = getattr(raw_image, 'n_frames', 1)
    if n_frames > 1 and foci_channel < n_frames:
        raw_image.seek(foci_channel)
        arr = np.array(raw_image)
        print(f"[threshold] Extracted frame {foci_channel}/{n_frames}: "
              f"dtype={arr.dtype}, min={arr.min()}, max={arr.max()}", file=sys.stderr)
        if arr.dtype != np.uint8:
            return normalize_to_uint8(arr)
        return arr

    # Multi-band
    bands = raw_image.split()
    if foci_channel < len(bands):
        arr = np.array(bands[foci_channel])
        if arr.dtype != np.uint8:
            return normalize_to_uint8(arr)
        return arr

    print(f"Warning: Could not extract channel {foci_channel}, using default",
          file=sys.stderr)
    arr = np.array(raw_image)
    if arr.dtype != np.uint8:
        return normalize_to_uint8(arr)
    return arr if arr.ndim == 2 else cv.cvtColor(arr, cv.COLOR_RGB2GRAY)


def process_foci_pipeline(gray_img, denoise_h, tophat_size, clip_limit, min_area):
    """Apply the foci isolation pipeline. Returns the processed image and binary foci mask."""
    # Ensure tophat_size is odd and >= 3
    tophat_size = max(3, tophat_size if tophat_size % 2 != 0 else tophat_size + 1)

    # Step 1: Denoise
    if denoise_h > 0:
        denoised = cv.fastNlMeansDenoising(gray_img, None, h=denoise_h,
                                            templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = gray_img.copy()

    # Step 2: Top-Hat (isolate small bright structures)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (tophat_size, tophat_size))
    foci_isolated = cv.morphologyEx(denoised, cv.MORPH_TOPHAT, kernel)

    # Step 3: CLAHE contrast enhancement
    clahe = cv.createCLAHE(clipLimit=float(max(0.1, clip_limit)), tileGridSize=(8, 8))
    enhanced = clahe.apply(foci_isolated)

    # Step 4: Area-based filtering
    _, binary_mask = cv.threshold(enhanced, 15, 255, cv.THRESH_BINARY)
    if min_area > 0:
        clean_mask_bool = morphology.remove_small_objects(binary_mask.astype(bool),
                                                          min_size=min_area)
        clean_mask = clean_mask_bool.astype(np.uint8) * 255
    else:
        clean_mask = binary_mask

    final = cv.bitwise_and(enhanced, enhanced, mask=clean_mask)

    return final, clean_mask


def count_foci_per_cell(foci_mask, cell_mask):
    """Count the number of distinct foci in each segmented cell.

    Args:
        foci_mask: Binary uint8 image (255 = foci pixel).
        cell_mask: Label image from segmentation (0 = background, >0 = cell label).

    Returns:
        dict mapping cell_label -> {foci_count, bbox}.
    """
    # Label connected components in the foci mask
    foci_labels = measure.label(foci_mask > 0, connectivity=2)

    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]

    results = {}
    for cell_label in cell_labels:
        cell_region = (cell_mask == cell_label)

        # Find bounding box
        ys, xs = np.where(cell_region)
        if len(ys) == 0:
            continue
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        # Count distinct foci overlapping this cell
        foci_in_cell = foci_labels[cell_region]
        foci_in_cell = foci_in_cell[foci_in_cell > 0]
        unique_foci = len(np.unique(foci_in_cell))

        results[int(cell_label)] = {
            "foci_count": unique_foci,
            "bbox": {
                "min_x": min_x, "min_y": min_y,
                "max_x": max_x, "max_y": max_y
            }
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Threshold-based Foci Classifier")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--foci_channel", type=int, default=-1)
    parser.add_argument("--denoise_h", type=int, default=15)
    parser.add_argument("--tophat_size", type=int, default=13)
    parser.add_argument("--clip_limit", type=float, default=6.0)
    parser.add_argument("--min_area", type=int, default=11)
    parser.add_argument("--damage_threshold", type=int, default=5)
    parser.add_argument("--save_foci_mask", action="store_true")
    args = parser.parse_args()

    # Extract foci channel
    try:
        gray = extract_channel(args.image_path, args.foci_channel)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load image: {str(e)}"}))
        sys.exit(1)

    # Load segmentation mask
    try:
        mask_img = Image.open(args.mask_path)
        cell_mask = np.array(mask_img)
        if cell_mask.ndim == 3:
            cell_mask = cell_mask[:, :, 0]
    except Exception as e:
        print(json.dumps({"error": f"Failed to load mask: {str(e)}"}))
        sys.exit(1)

    # Run foci isolation pipeline
    print(f"[threshold] Pipeline: denoise_h={args.denoise_h}, tophat={args.tophat_size}, "
          f"clip={args.clip_limit}, min_area={args.min_area}", file=sys.stderr)

    final_img, foci_mask = process_foci_pipeline(
        gray, args.denoise_h, args.tophat_size, args.clip_limit, args.min_area
    )

    # Save foci mask if requested
    foci_mask_path = None
    if args.save_foci_mask:
        base = os.path.splitext(args.mask_path)[0]
        # Save next to the original image, not the mask
        img_base = os.path.splitext(args.image_path)[0]
        foci_mask_path = img_base + "_foci_mask.png"
        cv.imwrite(foci_mask_path, foci_mask)
        print(f"[threshold] Saved foci mask: {foci_mask_path}", file=sys.stderr)

    # Count foci per cell
    cell_foci = count_foci_per_cell(foci_mask, cell_mask)

    if not cell_foci:
        output = {
            "cells": [],
            "summary": {"total": 0, "healthy": 0, "damaged": 0}
        }
        if foci_mask_path:
            output["foci_mask_path"] = foci_mask_path
        print(json.dumps(output))
        return

    # Classify based on damage threshold
    results = []
    for label, info in sorted(cell_foci.items()):
        foci_count = info["foci_count"]
        prediction = "Damaged" if foci_count >= args.damage_threshold else "Healthy"
        results.append({
            "label": label,
            "prediction": prediction,
            "foci_count": foci_count,
            "probability": foci_count / max(args.damage_threshold * 2, 1),
            "bbox": info["bbox"]
        })

    healthy_count = sum(1 for r in results if r["prediction"] == "Healthy")
    damaged_count = sum(1 for r in results if r["prediction"] == "Damaged")

    output = {
        "cells": results,
        "summary": {
            "total": len(results),
            "healthy": healthy_count,
            "damaged": damaged_count
        }
    }
    if foci_mask_path:
        output["foci_mask_path"] = foci_mask_path

    print(json.dumps(output))


if __name__ == "__main__":
    main()
