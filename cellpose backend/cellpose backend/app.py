from fastapi import FastAPI, UploadFile, File, Response, Query
import subprocess
import tempfile
import os
import shutil
import json
import logging
import sys
import time
import uuid
import threading
from pathlib import Path

# Setup Main Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CellposeAPI")

app = FastAPI()

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CELLPOSE_31_DIR = MODELS_DIR / "Cellpose 3.1"
CELLPOSE_SAM_DIR = MODELS_DIR / "CellposeSAM"
DINO_DIR = MODELS_DIR / "DINO"

is_windows = sys.platform.startswith("win")
venv_bin = "Scripts" if is_windows else "bin"
python_exec = "python.exe" if is_windows else "python"

VENV_V3 = BASE_DIR / "venv_v3" / venv_bin / python_exec
VENV_V4 = BASE_DIR / "venv_v4" / venv_bin / python_exec

ENV_MAPPING = {
    "Cellpose3.1": str(VENV_V3),
    "CellposeSAM": str(VENV_V4),
}

# --- MASK STORAGE (for /segment -> /classify reuse) ---
MASK_STORE_DIR = Path(tempfile.gettempdir()) / "cellpose_masks"
MASK_STORE_DIR.mkdir(parents=True, exist_ok=True)
MASK_TTL_SECONDS = 30 * 60  # 30 minutes
_mask_index_lock = threading.Lock()


def _cleanup_old_masks():
    """Delete mask folders older than MASK_TTL_SECONDS."""
    now = time.time()
    with _mask_index_lock:
        for child in MASK_STORE_DIR.iterdir():
            try:
                if child.is_dir() and (now - child.stat().st_mtime) > MASK_TTL_SECONDS:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass


@app.get("/getModels")
async def get_models():
    """Return available models grouped by type. DINO list is also included."""
    result = {"Cellpose3.1": [], "CellposeSAM": [], "DINO": []}
    for key, dir_path in (
        ("Cellpose3.1", CELLPOSE_31_DIR),
        ("CellposeSAM", CELLPOSE_SAM_DIR),
        ("DINO",        DINO_DIR),
    ):
        if dir_path.exists():
            try:
                for item in dir_path.iterdir():
                    result[key].append(item.name)
            except Exception as e:
                logger.error(f"Error scanning {key}: {e}")
    return result


@app.post("/segment")
async def segment(
        image: UploadFile = File(...),
        model_type: str = Query(..., enum=["Cellpose3.1", "CellposeSAM"]),
        model_name: str = Query(...),
        diameter: float = 0.0,
        channels: str = "0,0",
        use_gpu: bool = Query(False),
        batch_size: int = Query(64),
        resample: bool = Query(False),
        normalize: bool = Query(True),
        flow_threshold: float = Query(0.4),
        cellprob_threshold: float = Query(0.0),
        percentile_low: float = Query(1.0),
        percentile_high: float = Query(99.0),
        tile_norm: int = Query(0),
):
    _cleanup_old_masks()

    # Allocate a mask_id and a per-request directory; persist the input image there
    # so /classify can reuse it without a re-upload.
    mask_id = uuid.uuid4().hex
    mask_dir = MASK_STORE_DIR / mask_id
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original extension when possible (TIFF/PNG/JPG matter to classifiers).
    suffix = Path(image.filename or "image.png").suffix or ".png"
    image_path = mask_dir / f"image{suffix}"
    contents = await image.read()
    image_path.write_bytes(contents)

    # Worker also needs the file. Reuse the persisted copy as input.
    tmp_path = str(image_path)
    label_mask_path = mask_dir / "mask.png"

    try:
        python = ENV_MAPPING.get(model_type)
        if not python:
            return Response("Server Error: model_type misconfigured.", status_code=500)

        worker_path = os.path.join(os.path.dirname(__file__), "worker.py")

        cmd = [
            python, worker_path,
            "--image", tmp_path,
            "--model_type", model_type,
            "--model_name", model_name,
            "--diameter", str(diameter),
            "--channels", channels,
            "--batch_size", str(batch_size),
            "--flow_threshold", str(flow_threshold),
            "--cellprob_threshold", str(cellprob_threshold),
        ]
        if use_gpu:
            cmd.append("--use_gpu")
        if resample:
            cmd.append("--resample")
        if not normalize:
            cmd.append("--no_normalize")
        else:
            cmd.extend(["--percentile_low", str(percentile_low)])
            cmd.extend(["--percentile_high", str(percentile_high)])
            cmd.extend(["--tile_norm", str(tile_norm)])

        result = subprocess.check_output(cmd, timeout=600)
        output_json = json.loads(result.decode("utf-8"))
        if output_json.get("status") != "success":
            return Response(content=output_json.get("message", "Worker error"), status_code=500)

        # Reconstruct the label mask from outlines so /classify can reuse it.
        try:
            _reconstruct_label_mask_from_outlines(
                output_json["data"], image_path, label_mask_path
            )
        except Exception as e:
            logger.warning(f"Could not reconstruct label mask: {e}")

        headers = {"X-Mask-Id": mask_id}
        return Response(
            content=output_json["data"],
            media_type="application/json",
            headers=headers,
        )

    except subprocess.TimeoutExpired:
        return Response("Processing timed out.", status_code=504)
    except subprocess.CalledProcessError:
        logger.exception("Worker crashed")
        return Response("Internal Worker Error", status_code=500)


def _reconstruct_label_mask_from_outlines(data_json: str, image_path: Path, out_path: Path):
    """Build a uint16 label mask by filling each outline polygon with its cell ID."""
    import numpy as np
    import cv2
    data = json.loads(data_json)
    outlines = data.get("outlines", [])
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image
        pil = Image.open(image_path)
        h, w = pil.size[1], pil.size[0]
    else:
        h, w = img.shape[:2]

    label = np.zeros((h, w), dtype=np.uint16)
    for cell_id, outline_str in enumerate(outlines, start=1):
        if not outline_str:
            continue
        pts = [int(round(float(v))) for v in outline_str.split(",")]
        if len(pts) < 6:
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(label, [poly], int(cell_id))
    # PNG supports uint16 grayscale.
    cv2.imwrite(str(out_path), label)


@app.post("/classify")
async def classify(
        mask_id: str = Query(...),
        method:  str = Query("DINO", enum=["DINO", "Threshold"]),
        foci_channel: int = Query(-1),
        # DINO-specific
        model_name: str = Query("baseline_model.pth"),
        threshold:  float = Query(0.5),
        backbone_name: str = Query("vit_tiny_patch16_384"),
        image_size: int = Query(384),
        use_gpu: bool = Query(False),
        # Threshold-specific
        denoise_h: int = Query(15),
        tophat_size: int = Query(13),
        clip_limit: float = Query(6.0),
        min_area: int = Query(11),
        damage_threshold: int = Query(5),
):
    mask_dir = MASK_STORE_DIR / mask_id
    if not mask_dir.is_dir():
        return Response(f"Unknown mask_id: {mask_id}", status_code=404)

    images = [p for p in mask_dir.iterdir() if p.name.startswith("image")]
    if not images:
        return Response(f"Image for mask_id {mask_id} missing.", status_code=404)
    image_path = images[0]
    label_mask_path = mask_dir / "mask.png"
    if not label_mask_path.exists():
        return Response(
            "Label mask not available. Re-run segmentation with the latest backend.",
            status_code=500,
        )

    if method == "DINO":
        script = os.path.join(os.path.dirname(__file__), "classify_foci.py")
        model_path = DINO_DIR / model_name
        if not model_path.exists():
            return Response(f"DINO model not found: {model_path}", status_code=404)
        cmd = [
            str(VENV_V4), script,
            "--image_path", str(image_path),
            "--mask_path",  str(label_mask_path),
            "--model_path", str(model_path),
            "--backbone_name", backbone_name,
            "--image_size", str(image_size),
            "--threshold", str(threshold),
            "--foci_channel", str(foci_channel),
        ]
        if use_gpu:
            cmd.append("--use_gpu")
    else:  # Threshold
        script = os.path.join(os.path.dirname(__file__), "classify_foci_threshold.py")
        cmd = [
            str(VENV_V4), script,
            "--image_path", str(image_path),
            "--mask_path",  str(label_mask_path),
            "--foci_channel", str(foci_channel),
            "--denoise_h", str(denoise_h),
            "--tophat_size", str(tophat_size),
            "--clip_limit", str(clip_limit),
            "--min_area", str(min_area),
            "--damage_threshold", str(damage_threshold),
        ]

    try:
        result = subprocess.check_output(cmd, timeout=600)
        body = result.decode("utf-8").strip()
        # Some scripts also print info before the JSON; take the last JSON object.
        if not body.startswith("{"):
            last = body.rfind("\n{")
            if last >= 0:
                body = body[last + 1:]
        return Response(content=body, media_type="application/json")
    except subprocess.TimeoutExpired:
        return Response("Classification timed out.", status_code=504)
    except subprocess.CalledProcessError as e:
        logger.exception("Classifier failed")
        return Response(f"Classifier error: {e}", status_code=500)
