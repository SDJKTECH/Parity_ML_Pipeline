from __future__ import annotations
import os
import cv2
import numpy as np
from Pipeline.context import PipelineContext
from Pipeline.modules.bulbs.detector import BulbDetector

_BULB_DETECTOR: BulbDetector | None = None


def get_bulb_detector() -> BulbDetector:
  global _BULB_DETECTOR
  if _BULB_DETECTOR is None:
    _BULB_DETECTOR = BulbDetector()
  return _BULB_DETECTOR


def run(ctx: PipelineContext) -> bool:
  """STEP 3b: Bulb Detection & Coordinate Alignment."""
  # 1. Detect on raw frame to prevent blurred/warped edge degradation
  target_img = (
      ctx.raw_current_img
      if ctx.raw_current_img is not None
      else ctx.aligned_current_img
  )
  if target_img is None:
    return False

  detector = getattr(ctx, "bulb_engine", None) or get_bulb_detector()
  bulb_result = detector.detect_bulbs(target_img)

  raw_detections = bulb_result.get("detections", [])
  current_count = bulb_result.get("count", 0)
  baseline_count = ctx.master_data.get("bulb_count", 0)

  ctx.current_bulb_count = current_count
  ctx.baseline_bulb_count = baseline_count

  # 2. Project raw detection boxes onto the master aligned coordinate plane
  H = ctx.homography_matrix
  aligned_detections = []

  for d in raw_detections:
    d_copy = dict(d)
    raw_box = d.get("draw_bbox") or d.get("bbox", [])
    if len(raw_box) == 4 and H is not None:
      rx1, ry1, rx2, ry2 = raw_box
      corners = np.float32([
          [[rx1, ry1]],
          [[rx2, ry1]],
          [[rx2, ry2]],
          [[rx1, ry2]],
      ])
      warped_corners = cv2.perspectiveTransform(corners, H)
      wx1 = int(np.min(warped_corners[:, 0, 0]))
      wy1 = int(np.min(warped_corners[:, 0, 1]))
      wx2 = int(np.max(warped_corners[:, 0, 0]))
      wy2 = int(np.max(warped_corners[:, 0, 1]))
      d_copy["bbox_aligned"] = [wx1, wy1, wx2, wy2]
    else:
      d_copy["bbox_aligned"] = raw_box

    aligned_detections.append(d_copy)

  ctx.bulb_detections = aligned_detections

  print(
      f"[INFO] Active Bulbs Detected: {current_count} (Baseline:"
      f" {baseline_count})"
  )

  # 3. Export debug detection frame
  annotated_frame = bulb_result.get("annotated_frame")
  if annotated_frame is not None:
    out_dir = "output_images"
    os.makedirs(out_dir, exist_ok=True)
    room_tag = getattr(ctx, "room_name", "room")
    save_path = os.path.join(out_dir, f"{room_tag}_bulbs_annotated.jpg")
    cv2.imwrite(save_path, annotated_frame)
    print(
        f"[INFO] Bulb detection visual saved to: {save_path} ({current_count}"
        " bulb(s) marked)"
    )

  return True