from __future__ import annotations
import cv2
import numpy as np
from Pipeline.context import PipelineContext
from Pipeline.utils.features import extract_features


def run(ctx: PipelineContext) -> bool:
  """STEP 2: Feature Extraction on Raw Image with Homography Projection."""
  # Run YOLO on the unwarped raw capture to prevent 3D perspective distortion
  target_img = (
      ctx.raw_current_img
      if ctx.raw_current_img is not None
      else ctx.aligned_current_img
  )
  if target_img is None or ctx.model is None:
    ctx.halt = True
    return False

  # Extract raw furniture/object detections
  raw_data, _ = extract_features(target_img, ctx.model)
  H = ctx.homography_matrix

  projected_objects = []
  for obj in raw_data.get("objects", []):
    obj_copy = dict(obj)
    x1, y1, x2, y2 = obj["bbox"]

    if H is not None:
      # Project the 4 bbox corners into master baseline coordinate space
      corners = np.float32([
          [[x1, y1]],
          [[x2, y1]],
          [[x2, y2]],
          [[x1, y2]],
      ])
      warped = cv2.perspectiveTransform(corners, H)

      wx1 = float(np.min(warped[:, 0, 0]))
      wy1 = float(np.min(warped[:, 0, 1]))
      wx2 = float(np.max(warped[:, 0, 0]))
      wy2 = float(np.max(warped[:, 0, 1]))

      obj_copy["bbox"] = [
          round(wx1, 2),
          round(wy1, 2),
          round(wx2, 2),
          round(wy2, 2),
      ]
      obj_copy["centroid"] = (int((wx1 + wx2) / 2), int((wy1 + wy2) / 2))

    projected_objects.append(obj_copy)

  ctx.current_data = {
      "image_shape": raw_data.get("image_shape"),
      "brightness": raw_data.get("brightness"),
      "objects": projected_objects,
  }

  return True