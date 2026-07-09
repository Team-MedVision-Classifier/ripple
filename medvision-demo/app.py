"""
MedVision Demo - Cell Segmentation & Foci Classification
A Gradio app for Cellpose-based cell segmentation and threshold-based foci classification.
Designed to run on HuggingFace Spaces.
"""

import os
import tempfile

import cv2 as cv
import gradio as gr
import numpy as np
from cellpose import models
from PIL import Image
from skimage import measure, morphology


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile-based normalization to uint8."""
    arr = arr.astype(np.float64)
    p_low, p_high = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if p_high - p_low < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def load_image(path: str):
    """Load an image, normalize high-bit-depth to uint8, return (gray, rgb)."""
    raw = Image.open(path)
    arr = np.array(raw)
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        gray = arr
        rgb = cv.cvtColor(arr, cv.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    else:
        gray = arr.squeeze()
        rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    return gray, rgb


def golden_angle_colormap(n: int = 256):
    """Generate distinct colors using the golden-angle method (matches Java UI)."""
    colors = np.zeros((n, 3), dtype=np.uint8)
    golden = 137.508
    for i in range(n):
        hue = (i * golden) % 360
        h = hue / 2  # OpenCV hue range 0-179
        hsv = np.uint8([[[int(h), 200, 220]]])
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        colors[i] = bgr[0, 0, ::-1]  # BGR -> RGB
    return colors


COLORMAP = golden_angle_colormap()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a label mask to an RGB image using the golden-angle colormap."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(mask)
    for lab in labels:
        if lab == 0:
            continue
        color = COLORMAP[lab % len(COLORMAP)]
        rgb[mask == lab] = color
    return rgb


def blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Blend overlay onto base where overlay is non-black."""
    mask = overlay.sum(axis=2) > 0
    out = base.copy()
    out[mask] = cv.addWeighted(base, 1 - alpha, overlay, alpha, 0)[mask]
    return out


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def run_segmentation(
    image,
    model_type: str,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    use_gpu: bool,
):
    """Run Cellpose segmentation and return results."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    gray, rgb = load_image(image)

    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    masks, flows, styles, diams = model.eval(
        gray,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],
    )

    num_cells = masks.max()
    colored = colorize_mask(masks)
    overlay = blend_overlay(rgb, colored, 0.45)

    # Compute per-cell properties
    props = measure.regionprops(masks)
    table_rows = []
    for p in props:
        table_rows.append([
            p.label,
            p.area,
            round(p.equivalent_diameter_area, 1),
            round(p.eccentricity, 3),
            round(p.centroid[1], 1),
            round(p.centroid[0], 1),
        ])

    status = f"Detected **{num_cells}** cells"
    if diams is not None:
        status += f" | Auto diameter: {diams:.1f} px"

    # Save mask to temp file for classification step
    tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
    Image.fromarray(masks.astype(np.uint16)).save(tmp.name)
    tmp.close()

    return overlay, colored, status, table_rows, tmp.name


# ---------------------------------------------------------------------------
# Foci Classification (threshold-based)
# ---------------------------------------------------------------------------

def process_foci_pipeline(gray_img, denoise_h, tophat_size, clip_limit, min_area):
    """Morphological pipeline to isolate foci. Returns (processed, binary_mask)."""
    tophat_size = max(3, tophat_size if tophat_size % 2 != 0 else tophat_size + 1)

    if denoise_h > 0:
        denoised = cv.fastNlMeansDenoising(gray_img, None, h=denoise_h,
                                            templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = gray_img.copy()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (tophat_size, tophat_size))
    foci_isolated = cv.morphologyEx(denoised, cv.MORPH_TOPHAT, kernel)

    clahe = cv.createCLAHE(clipLimit=float(max(0.1, clip_limit)), tileGridSize=(8, 8))
    enhanced = clahe.apply(foci_isolated)

    _, binary_mask = cv.threshold(enhanced, 15, 255, cv.THRESH_BINARY)
    if min_area > 0:
        clean = morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_area)
        binary_mask = clean.astype(np.uint8) * 255

    return enhanced, binary_mask


def run_classification(
    image,
    mask_path: str,
    denoise_h: int,
    tophat_size: int,
    clip_limit: float,
    min_area: int,
    damage_threshold: int,
):
    """Run threshold-based foci classification on segmented cells."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    if not mask_path or not os.path.exists(mask_path):
        raise gr.Error("Run segmentation first before classifying.")

    gray, rgb = load_image(image)
    cell_mask = np.array(Image.open(mask_path))
    if cell_mask.ndim == 3:
        cell_mask = cell_mask[:, :, 0]

    img_h, img_w = gray.shape
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]

    if len(cell_labels) == 0:
        raise gr.Error("No cells found in the segmentation mask.")

    classification_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    foci_vis = np.zeros((img_h, img_w), dtype=np.uint8)
    table_rows = []

    for lab in cell_labels:
        ys, xs = np.where(cell_mask == lab)
        if len(ys) == 0:
            continue
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        pad_x = max(int((max_x - min_x) * 0.1), 2)
        pad_y = max(int((max_y - min_y) * 0.1), 2)
        x0, y0 = max(0, min_x - pad_x), max(0, min_y - pad_y)
        x1, y1 = min(img_w, max_x + pad_x + 1), min(img_h, max_y + pad_y + 1)

        cell_patch = gray[y0:y1, x0:x1].copy()
        cell_region = cell_mask[y0:y1, x0:x1]
        cell_only = (cell_region == lab).astype(np.uint8) * 255
        cell_patch = cv.bitwise_and(cell_patch, cell_patch, mask=cell_only)

        _, patch_foci = process_foci_pipeline(cell_patch, denoise_h, tophat_size, clip_limit, min_area)
        patch_foci = cv.bitwise_and(patch_foci, cell_only)

        foci_labels = measure.label(patch_foci > 0, connectivity=2)
        foci_count = foci_labels.max()

        foci_vis[y0:y1, x0:x1] = np.maximum(foci_vis[y0:y1, x0:x1], patch_foci)

        prediction = "Damaged" if foci_count >= damage_threshold else "Healthy"
        color = (220, 50, 50) if prediction == "Damaged" else (50, 200, 50)
        classification_overlay[cell_mask == lab] = color

        table_rows.append([
            int(lab), prediction, int(foci_count),
        ])

    overlay = blend_overlay(rgb, classification_overlay, 0.45)

    # Foci mask visualization (cyan)
    foci_rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    foci_rgb[foci_vis > 0] = (0, 220, 220)
    foci_overlay = blend_overlay(rgb, foci_rgb, 0.6)

    healthy = sum(1 for r in table_rows if r[1] == "Healthy")
    damaged = sum(1 for r in table_rows if r[1] == "Damaged")
    total = len(table_rows)
    status = f"**{total}** cells | **{healthy}** Healthy | **{damaged}** Damaged"

    return overlay, foci_overlay, status, table_rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="MedVision - Cell Segmentation & Foci Classification",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # MedVision Demo
        **Cell Segmentation & Foci Classification**

        Upload a microscopy image, run Cellpose segmentation, then classify cells
        as Healthy or Damaged using threshold-based foci detection.
        """
    )

    # Hidden state to pass mask path between tabs
    mask_state = gr.State(value=None)

    with gr.Tabs():
        # ---- Tab 1: Segmentation ----
        with gr.Tab("Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload Microscopy Image",
                        type="filepath",
                        height=200,
                    )
                    model_type = gr.Dropdown(
                        choices=["cyto3", "cyto2", "cyto", "nuclei"],
                        value="cyto3",
                        label="Cellpose Model",
                    )
                    diameter = gr.Slider(
                        0, 500, value=30, step=1,
                        label="Cell Diameter (px) - 0 for auto",
                    )
                    flow_threshold = gr.Slider(
                        0.0, 3.0, value=0.4, step=0.05,
                        label="Flow Threshold",
                    )
                    cellprob_threshold = gr.Slider(
                        -6.0, 6.0, value=0.0, step=0.1,
                        label="Cell Probability Threshold",
                    )
                    use_gpu = gr.Checkbox(label="Use GPU", value=False)
                    seg_btn = gr.Button("Run Segmentation", variant="primary")

                with gr.Column(scale=2):
                    seg_status = gr.Markdown("Upload an image and click **Run Segmentation**.")
                    with gr.Row():
                        seg_overlay = gr.Image(label="Segmentation Overlay", interactive=False)
                        seg_mask_img = gr.Image(label="Cell Mask", interactive=False)
                    seg_table = gr.Dataframe(
                        headers=["Label", "Area (px)", "Diameter", "Eccentricity", "X", "Y"],
                        label="Cell Properties",
                        interactive=False,
                    )

            seg_btn.click(
                fn=run_segmentation,
                inputs=[img_input, model_type, diameter, flow_threshold, cellprob_threshold, use_gpu],
                outputs=[seg_overlay, seg_mask_img, seg_status, seg_table, mask_state],
            )

        # ---- Tab 2: Foci Classification ----
        with gr.Tab("Foci Classification"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Threshold Pipeline Parameters**")
                    denoise_h = gr.Slider(0, 50, value=15, step=1, label="Denoise Strength (h)")
                    tophat_size = gr.Slider(3, 51, value=13, step=2, label="Top-Hat Kernel Size")
                    clip_limit = gr.Slider(0.1, 20.0, value=6.0, step=0.1, label="CLAHE Clip Limit")
                    min_area = gr.Slider(0, 500, value=11, step=1, label="Min Foci Area (px)")
                    damage_threshold = gr.Slider(1, 100, value=5, step=1, label="Damage Threshold (foci count)")
                    cls_btn = gr.Button("Run Classification", variant="primary")

                with gr.Column(scale=2):
                    cls_status = gr.Markdown("Run segmentation first, then classify cells here.")
                    with gr.Row():
                        cls_overlay = gr.Image(label="Classification Overlay", interactive=False)
                        foci_overlay = gr.Image(label="Foci Detection", interactive=False)
                    cls_table = gr.Dataframe(
                        headers=["Label", "Prediction", "Foci Count"],
                        label="Classification Results",
                        interactive=False,
                    )

            cls_btn.click(
                fn=run_classification,
                inputs=[img_input, mask_state, denoise_h, tophat_size, clip_limit, min_area, damage_threshold],
                outputs=[cls_overlay, foci_overlay, cls_status, cls_table],
            )

    gr.Markdown(
        """
        ---
        **How it works:**
        1. **Segmentation** uses [Cellpose](https://github.com/MouseLand/cellpose) to detect individual cells.
        2. **Classification** applies a morphological pipeline (NLM Denoising -> Top-Hat -> CLAHE -> Area Filtering)
           to detect foci per cell, then classifies as Healthy/Damaged based on foci count.

        Green = Healthy | Red = Damaged | Cyan = Detected Foci
        """
    )

if __name__ == "__main__":
    demo.launch()
