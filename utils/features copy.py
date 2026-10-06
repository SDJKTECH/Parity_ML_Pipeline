import cv2
import sys
from .brightness import calculate_brightness

def extract_features(image_path, model):
    """Extracts bounding boxes, centroids, and lighting data."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    # Ran YOLO model on the image to detect objects
    results = model(image_path, verbose=False)
    objects = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            coords = box.xyxy[0].tolist()  
            
            centroid_x = int((coords[0] + coords[2]) / 2)
            centroid_y = int((coords[1] + coords[3]) / 2)

            objects.append({
                "label": class_name,
                "bbox": [round(c, 2) for c in coords],
                "centroid": (centroid_x, centroid_y),
                "confidence": round(float(box.conf[0]), 2)
            })

    return {
        "image_shape": img.shape[:2],
        "brightness": round(calculate_brightness(img), 2),
        "objects": objects
    }, img