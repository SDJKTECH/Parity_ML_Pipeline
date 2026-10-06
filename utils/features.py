from __future__ import annotations
import cv2
import numpy as np
from .brightness import calculate_brightness


def extract_features(
    image_input: str | np.ndarray,
    model,
    conf_threshold: float = 0.50,
) -> tuple[dict, np.ndarray]:
    """
    Extracts bounding boxes, centroids, and lighting data from an image path or array.

    Args:
        image_input: Filepath string OR already loaded OpenCV BGR numpy array.
        model: Loaded Ultralytics YOLO model instance.
        conf_threshold: Minimum confidence score to register an object.

    Returns:
        tuple[dict, np.ndarray]: Metadata dictionary and the loaded BGR image.
    """
    # 1. Handle string path vs in-memory numpy array
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if img.size == 0:
        raise ValueError("Provided image array is empty.")

    # Ensure memory is contiguous and uint8
    if not img.flags.c_contiguous:
        img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]

    # 2. Run YOLO directly on the array with explicit confidence threshold
    results = model(img, conf=conf_threshold, verbose=False)
    objects = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            coords = box.xyxy[0].tolist()

            # Clamp bounding box coordinates strictly inside image frame
            x1 = max(0, min(w - 1, coords[0]))
            y1 = max(0, min(h - 1, coords[1]))
            x2 = max(0, min(w - 1, coords[2]))
            y2 = max(0, min(h - 1, coords[3]))

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            objects.append({
                "label": str(class_name),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "centroid": (centroid_x, centroid_y),
                "confidence": round(float(box.conf[0]), 2),
            })

    metadata = {
        "image_shape": [h, w],
        "brightness": round(calculate_brightness(img), 2),
        "objects": objects,
    }

    return metadata, img