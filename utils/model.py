from ultralytics import YOLO

def load_yolo(model_name="yolov8l.pt"):
    """Loads lightweight pre-trained YOLO model for inventory and clutter detection."""
    return YOLO(model_name)