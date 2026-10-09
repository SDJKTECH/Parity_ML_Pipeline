"""Local Master Registration Engine (Prompt-Routed)

Processes owner golden reference images locally on disk:
- Inspects prompt intent (BULBS vs. OBJECTS)
- Extracts ONLY the requested features based on prompt mode
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
from Pipeline.utils.intent import classify_prompt_mode
from Pipeline.modules.bulbs.detector import BulbDetector
from Pipeline.config import MASTER_DIR, BASELINES_DIR, VALID_EXTENSIONS

# Lazy-loaded singletons to avoid import-time bottlenecks
_BULB_ENGINE: BulbDetector | None = None
_YOLO_MODEL = None


def get_engines() -> BulbDetector:
    global _BULB_ENGINE
    if _BULB_ENGINE is None:
        print("[INIT] Loading Bulb Detector & CLIP VLM weights into memory...")
        _BULB_ENGINE = BulbDetector()
    return _BULB_ENGINE


def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        print("[INIT] Loading YOLO model weights into memory...")
        _YOLO_MODEL = load_yolo("yolov8x.pt")
    return _YOLO_MODEL


def process_master_image(
    image_input: str | np.ndarray | bytes,
    room_name: str,
    prompt_name: str | None = None,
    prompt_instructions: str | None = None,
    output_baseline_dir: str = BASELINES_DIR,
    export_annotated_dir: str | None = "Testing",
) -> dict:
    """Processes a single owner room image and writes prompt-routed baseline artifacts.

    Args:
        image_input: File path string, raw byte buffer, or loaded cv2 BGR image.
        room_name: Identifier for the space/room (e.g. "Table 1", "FirstFloor").
        prompt_name: Display label of the inspection preset.
        prompt_instructions: Specific audit guidance given by the owner.
        output_baseline_dir: Target directory for baseline artifacts.
        export_annotated_dir: Optional directory to export detection visuals.

    Returns:
        dict: The full metadata payload saved to baseline JSON.
    """
    os.makedirs(output_baseline_dir, exist_ok=True)

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

    h, w = img.shape[:2]

    # 2. Classify prompt intent
    p_name = prompt_name or "General Room Inspection"
    p_instructions = (
        prompt_instructions
        or "Inspect room state against expected baseline inventory and lighting."
    )
    combined_prompt_text = f"{p_name} {p_instructions}"
    mode = classify_prompt_mode(combined_prompt_text)

    print(f"\n[MASTER REGISTRATION] Target Space : {room_name}")
    print(f"[MASTER REGISTRATION] Prompt Intent: >>> {mode} <<<")

    # 3. Base metadata structure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = round(float(np.mean(gray)), 2)

    data = {
        "room_name": room_name,
        "image_shape": [h, w],
        "brightness": avg_brightness,
        "audit_mode": mode,
        "prompt_name": p_name,
        "prompt_instructions": p_instructions,
        "objects": [],
        "bulb_count": 0,
        "bulb_detections": [],
        "bulbs": [],
    }

    annotated_frame = None

    # 4. Conditionally execute only the pipeline branch specified by the prompt
    if mode == "OBJECTS":
        print("[EXTRACTOR] Running YOLO furniture & inventory baseline extraction...")
        yolo_model = get_yolo_model()
        features_data, _ = extract_features(img, yolo_model)
        data["objects"] = features_data.get("objects", [])
        data["bulb_count"] = 0
        data["bulb_detections"] = []
        data["bulbs"] = []

    elif mode == "BULBS":
        print("[EXTRACTOR] Running Bulb Detector & CLIP baseline extraction...")
        bulb_engine = get_engines()
        bulb_res = bulb_engine.detect_bulbs(img)
        data["objects"] = []
        data["bulb_count"] = bulb_res.get("count", 0)
        data["bulb_detections"] = bulb_res.get("detections", [])
        data["bulbs"] = bulb_res.get("detections", [])
        annotated_frame = bulb_res.get("annotated_frame")

    else:
        # Fallback: extract both if prompt is ambiguous
        print("[EXTRACTOR] Running full dual extraction (Bulbs + Objects)...")
        yolo_model = get_yolo_model()
        bulb_engine = get_bulb_engine()
        features_data, _ = extract_features(img, yolo_model)
        bulb_res = bulb_engine.detect_bulbs(img)

        data["objects"] = features_data.get("objects", [])
        data["bulb_count"] = bulb_res.get("count", 0)
        data["bulb_detections"] = bulb_res.get("detections", [])
        data["bulbs"] = bulb_res.get("detections", [])
        annotated_frame = bulb_res.get("annotated_frame")

    # 5. Save local baseline artifacts
    ref_out_path = os.path.join(output_baseline_dir, f"{room_name}_ref.jpg")
    json_out_path = os.path.join(output_baseline_dir, f"{room_name}_baseline.json")

    cv2.imwrite(ref_out_path, img)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # 6. Optional: Save visual debug check
    if export_annotated_dir and annotated_frame is not None:
        os.makedirs(export_annotated_dir, exist_ok=True)
        ann_out_path = os.path.join(export_annotated_dir, f"{room_name}_annotated.jpg")
        cv2.imwrite(ann_out_path, annotated_frame)

    print(
        f"[SUCCESS] Baseline created for '{room_name}' "
        f"(Mode: {mode} | Objects: {len(data['objects'])} | Bulbs: {data['bulb_count']})"
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