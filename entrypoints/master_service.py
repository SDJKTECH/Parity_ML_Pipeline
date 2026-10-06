"""Local Master Registration Engine (No Cloud / S3 dependencies)

Processes owner golden reference images locally on disk:
- Extracts structural features and active light source coordinates
- Saves baselines/<room>_baseline.json with owner prompt rules
- Saves baselines/<room>_ref.jpg
- Optionally exports annotated visual detections to Testing/<room>_annotated.jpg
"""

import os
import json
import cv2
import numpy as np

from Pipeline.utils.model import load_yolo
from Pipeline.utils.features import extract_features
from Pipeline.modules.bulbs.detector import BulbDetector
from Pipeline.config import MASTER_DIR, BASELINES_DIR, VALID_EXTENSIONS

# Lazy-loaded singletons to avoid import-time bottlenecks
_BULB_ENGINE: BulbDetector | None = None
_YOLO_MODEL = None


def get_engines() -> tuple[BulbDetector, any]:
    """Initializes or retrieves cached detector models."""
    global _BULB_ENGINE, _YOLO_MODEL

    if _BULB_ENGINE is None:
        print("[INIT] Loading Bulb Detector & CLIP VLM weights into memory...")
        _BULB_ENGINE = BulbDetector()

    if _YOLO_MODEL is None:
        print("[INIT] Loading YOLO model weights into memory...")
        _YOLO_MODEL = load_yolo("yolov8l.pt")

    return _BULB_ENGINE, _YOLO_MODEL


def process_master_image(
    image_input: str | np.ndarray | bytes,
    room_name: str,
    prompt_name: str | None = None,
    prompt_instructions: str | None = None,
    output_baseline_dir: str = BASELINES_DIR,
    export_annotated_dir: str | None = "Testing",
) -> dict:
    """Processes a single owner room image and writes local baseline artifacts.

    Args:
        image_input: File path string, raw byte buffer, or loaded cv2 BGR image.
        room_name: Identifier for the space/room (e.g. "Table 1", "FirstFloor").
        prompt_name: Display label of the inspection preset (e.g. "Staff Area Cleanliness Check").
        prompt_instructions: Specific audit guidance given by the owner.
        output_baseline_dir: Target directory for baseline artifacts.
        export_annotated_dir: Optional directory to export detection visuals.

    Returns:
        dict: The full metadata payload saved to baseline JSON.
    """
    os.makedirs(output_baseline_dir, exist_ok=True)
    bulb_engine, model = get_engines()

    # 1. Load image based on input type
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Master image not found: {image_input}")
        img = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        img = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = image_input

    if img is None or img.size == 0:
        raise ValueError(f"Could not load valid image for room: {room_name}")

    # 2. Extract keypoints and furniture baseline data (direct in-memory)
    data, _ = extract_features(img, model)

    # 3. Detect and verify baseline active light sources
    bulb_res = bulb_engine.detect_bulbs(img)
    data["room_name"] = room_name
    data["bulb_count"] = bulb_res.get("count", 0)
    data["bulb_detections"] = bulb_res.get("detections", [])
    data["bulbs"] = bulb_res.get("detections", [])  # Backward-compatible alias
    data["image_shape"] = list(img.shape[:2])  # [height, width] for scale normalizing

    # 4. Attach Owner's Custom Prompt Rules
    data["prompt_name"] = prompt_name or "General Room Inspection"
    data["prompt_instructions"] = (
        prompt_instructions
        or "Inspect room state against expected baseline inventory and lighting."
    )

    # 5. Save local baseline artifacts
    ref_out_path = os.path.join(output_baseline_dir, f"{room_name}_ref.jpg")
    json_out_path = os.path.join(output_baseline_dir, f"{room_name}_baseline.json")

    cv2.imwrite(ref_out_path, img)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # 6. Optional: Save annotated visual check locally
    if export_annotated_dir and bulb_res.get("annotated_frame") is not None:
        os.makedirs(export_annotated_dir, exist_ok=True)
        ann_out_path = os.path.join(export_annotated_dir, f"{room_name}_annotated.jpg")
        cv2.imwrite(ann_out_path, bulb_res["annotated_frame"])

    print(
        f"[SUCCESS] Baseline created for '{room_name}' "
        f"(Active Bulbs: {data['bulb_count']} | Prompt: '{data['prompt_name']}')"
    )
    return data


def generate_all_baselines(master_dir: str = MASTER_DIR):
    """Batch processes all master images found in the local master_images folder."""
    if not os.path.isdir(master_dir):
        print(f"[ERROR] Directory '{master_dir}' does not exist.")
        return

    images = [f for f in os.listdir(master_dir) if f.lower().endswith(VALID_EXTENSIONS)]
    print(f"[INFO] Found {len(images)} master images in '{master_dir}' to process.")

    for img_name in images:
        room_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(master_dir, img_name)
        process_master_image(image_input=img_path, room_name=room_name)


if __name__ == "__main__":
    generate_all_baselines()