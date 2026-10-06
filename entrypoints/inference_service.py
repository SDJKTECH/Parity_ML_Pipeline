"""Worker Inference Service

Audits runtime room captures against registered baselines and paints visual
discrepancy annotations.
"""

from __future__ import annotations
import json
import os
import cv2
import numpy as np

from Pipeline.config import BASELINES_DIR, OUTPUT_DIR
from Pipeline.context import PipelineContext
from Pipeline.runner import PipelineRunner
from Pipeline.utils.model import load_yolo

_RUNNER: PipelineRunner | None = None
_YOLO_MODEL = None


def get_inference_engines():
  global _RUNNER, _YOLO_MODEL
  if _RUNNER is None:
    print("[INIT] Loading PipelineRunner engine...")
    _RUNNER = PipelineRunner()
  if _YOLO_MODEL is None:
    print("[INIT] Loading YOLO furniture model into memory...")
    _YOLO_MODEL = load_yolo("yolov8l.pt")
  return _RUNNER, _YOLO_MODEL


def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
  """Calculates Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  inter_area = max(0, xB - xA) * max(0, yB - yA)
  boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  union = float(boxA_area + boxB_area - inter_area)
  return inter_area / union if union > 0 else 0.0


def draw_discrepancy_annotations(ctx: PipelineContext) -> np.ndarray:
  """Paints detections onto the canvas selectively based on ctx.audit_mode."""
  canvas = (
      ctx.aligned_current_img.copy()
      if ctx.aligned_current_img is not None
      else (
          ctx.raw_current_img.copy()
          if ctx.raw_current_img is not None
          else None
      )
  )
  if canvas is None:
    return np.zeros((480, 640, 3), dtype=np.uint8)

  h, w = canvas.shape[:2]
  gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
  mode = getattr(ctx, "audit_mode", "ALL")
  hud_items = []

  # 1. Obtain visibility mask (valid pixels vs black warp border)
  if ctx.valid_mask is not None:
    valid_mask = ctx.valid_mask
  else:
    _, valid_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)

  # 2. Scale factor calculation (if master baseline resolution differs)
  master_shape = ctx.master_data.get("image_shape")
  scale_x, scale_y = 1.0, 1.0
  if master_shape and len(master_shape) >= 2:
    orig_h, orig_w = master_shape[0], master_shape[1]
    if orig_w > 0 and orig_h > 0:
      scale_x = w / float(orig_w)
      scale_y = h / float(orig_h)

  # =========================================================================
  # BRANCH A: BULB VISUAL ANNOTATIONS (Only when mode is 'BULBS' or 'ALL')
  # =========================================================================
  if mode in ("BULBS", "ALL"):
    active_boxes = []
    for bulb in ctx.bulb_detections:
      bbox = (
          bulb.get("bbox_aligned") or bulb.get("draw_bbox") or bulb.get("bbox")
      )
      if bbox and len(bbox) == 4:
        bx1, by1, bx2, by2 = [
            max(0, min(dim, int(v)))
            for dim, v in zip([w - 1, h - 1, w - 1, h - 1], bbox)
        ]
        if bx2 > bx1 and by2 > by1:
          active_boxes.append({
              "box": [bx1, by1, bx2, by2],
              "conf": bulb.get("confidence", 1.0),
              "matched": False,
          })

    expected_bulbs = ctx.master_data.get(
        "bulb_detections"
    ) or ctx.master_data.get("bulbs", [])
    out_of_view_count = 0
    lights_off_count = 0

    for b_idx, m_bulb in enumerate(expected_bulbs, start=1):
      m_box = m_bulb.get("bbox", [])
      if len(m_box) != 4:
        continue

      mx1 = int(m_box[0] * scale_x)
      my1 = int(m_box[1] * scale_y)
      mx2 = int(m_box[2] * scale_x)
      my2 = int(m_box[3] * scale_y)

      mx1, my1 = max(0, min(w - 1, mx1)), max(0, min(h - 1, my1))
      mx2, my2 = max(0, min(w - 1, mx2)), max(0, min(h - 1, my2))
      if mx2 <= mx1 or my2 <= my1:
        continue

      master_box = [mx1, my1, mx2, my2]

      # Check visible ratio against black border
      bulb_mask_patch = valid_mask[my1:my2, mx1:mx2]
      visible_ratio = (
          np.count_nonzero(bulb_mask_patch) / bulb_mask_patch.size
          if bulb_mask_patch.size > 0
          else 0
      )

      if visible_ratio < 0.35:
        out_of_view_count += 1
        continue

      # Spatial & IoU proximity matching against active green boxes
      m_cx, m_cy = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0
      matched_active = False

      for act in active_boxes:
        iou = calculate_iou(master_box, act["box"])
        ax1, ay1, ax2, ay2 = act["box"]
        a_cx, a_cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        dist = np.hypot(m_cx - a_cx, m_cy - a_cy)

        if iou > 0.08 or dist < 110:
          matched_active = True
          act["matched"] = True
          break

      # Expanded Window Local Luminance Verification (±25px)
      if not matched_active:
        search_pad = 25
        sx1, sy1 = max(0, mx1 - search_pad), max(0, my1 - search_pad)
        sx2, sy2 = min(w, mx2 + search_pad), min(h, my2 + search_pad)

        patch = gray_canvas[sy1:sy2, sx1:sx2]
        patch_mask = valid_mask[sy1:sy2, sx1:sx2]
        valid_patch = patch[patch_mask > 0] if patch_mask.size > 0 else patch

        if valid_patch.size > 0:
          if (
              int(np.max(valid_patch)) >= 200
              and float(np.percentile(valid_patch, 95.0)) >= 170
          ):
            matched_active = True

      # Truly off fixture in view
      if not matched_active:
        lights_off_count += 1
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
        label_txt = f"LIGHT OFF #{b_idx}"
        (lw, lh), _ = cv2.getTextSize(
            label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(
            canvas,
            (mx1, max(0, my1 - lh - 4)),
            (mx1 + lw + 4, my1),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            canvas,
            label_txt,
            (mx1 + 2, max(12, my1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw active green bulbs
    for act in active_boxes:
      x1, y1, x2, y2 = act["box"]
      conf = act["conf"]
      cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(
          canvas,
          f"Bulb ON ({conf*100:.0f}%)",
          (x1, max(16, y1 - 4)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.42,
          (0, 255, 0),
          1,
          cv2.LINE_AA,
      )

    if lights_off_count > 0:
      hud_items.append(f"LIGHTS OFF: {lights_off_count} inactive")
    if out_of_view_count > 0:
      hud_items.append(
          f"OUT OF VIEW: {out_of_view_count} fixture(s) not in angle"
      )

  # =========================================================================
  # BRANCH B: OBJECT DISCREPANCIES (Only when mode is 'OBJECTS' or 'ALL')
  # =========================================================================
  if mode in ("OBJECTS", "ALL"):
    for missing in set(ctx.missing_items):
      hud_items.append(f"MISSING: {missing}")
    for drift in ctx.drift_alerts:
      hud_items.append(f"DRIFT: {drift}")
    for clutter in set(ctx.clutter_items):
      hud_items.append(f"CLUTTER: {clutter}")

  # =========================================================================
  # 3. RENDER HUD BADGE
  # =========================================================================
  if hud_items:
    badge_w = min(w - 20, 420)
    badge_h = 35 + len(hud_items) * 22
    overlay = canvas.copy()
    cv2.rectangle(
        overlay, (10, 10), (10 + badge_w, 10 + badge_h), (20, 20, 20), -1
    )
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    verdict_color = (0, 0, 255) if ctx.verdict == "REJECTED" else (0, 255, 0)
    cv2.putText(
        canvas,
        f"VERDICT: {ctx.verdict} ({mode} MODE)",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        verdict_color,
        2,
        cv2.LINE_AA,
    )

    for idx, item in enumerate(hud_items):
      cv2.putText(
          canvas,
          f"! {item}",
          (20, 56 + idx * 22),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.44,
          (255, 255, 255),
          1,
          cv2.LINE_AA,
      )

  return canvas


def evaluate_worker_capture(
    current_image_input: str | np.ndarray | bytes,
    room_name: str,
    prompt_text: str | None = None,
    audit_mode: str = "ALL",
    baseline_dir: str = BASELINES_DIR,
    output_dir: str = OUTPUT_DIR,
) -> dict:
  """Audits worker captures with dynamic prompt intent routing."""
  runner, model = get_inference_engines()

  json_path = os.path.join(baseline_dir, f"{room_name}_baseline.json")
  ref_img_path = os.path.join(baseline_dir, f"{room_name}_ref.jpg")

  if not os.path.exists(json_path) or not os.path.exists(ref_img_path):
    raise FileNotFoundError(
        f"Missing baseline files for room '{room_name}' in {baseline_dir}"
    )

  with open(json_path, "r", encoding="utf-8") as f:
    master_data = json.load(f)

  master_img = cv2.imread(ref_img_path)

  # Handle file path, raw bytes, or numpy array
  if isinstance(current_image_input, str):
    raw_current_img = cv2.imread(current_image_input)
    filename = os.path.basename(current_image_input)
    img_path = current_image_input
  elif isinstance(current_image_input, bytes):
    raw_current_img = cv2.imdecode(
        np.frombuffer(current_image_input, np.uint8), cv2.IMREAD_COLOR
    )
    filename = f"{room_name}_capture.jpg"
    img_path = "in_memory"
  else:
    raw_current_img = current_image_input
    filename = f"{room_name}_capture.jpg"
    img_path = "in_memory"

  if raw_current_img is None or raw_current_img.size == 0:
    raise ValueError(
        f"Could not load valid image from input: {current_image_input}"
    )

  # Build context with prompt and mode routing
  ctx = PipelineContext(
      room_name=room_name,
      raw_current_img=raw_current_img,
      master_img=master_img,
      master_data=master_data,
      model=model,
      img_path=img_path,
      filename=filename,
      prompt_text=prompt_text,
      audit_mode=audit_mode,
  )

  # Run pipeline stages
  ctx = runner.run(ctx)

  # Paint visual annotations
  annotated_canvas = draw_discrepancy_annotations(ctx)

  os.makedirs(output_dir, exist_ok=True)
  out_path = os.path.join(output_dir, f"{room_name}_annotated_result.jpg")
  cv2.imwrite(out_path, annotated_canvas)
  print(f"[SAVED] Inspection visual written to: {out_path}")
  print(f"[VERDICT] Status: {ctx.verdict} (Mode: {ctx.audit_mode})")

  l_delta = getattr(ctx, "lighting_delta", 0.0) or getattr(
      ctx, "brightness_diff", 0.0
  )

  return {
      "room_name": ctx.room_name,
      "audit_mode": ctx.audit_mode,
      "verdict": ctx.verdict,
      "ssim_score": ctx.ssim_score,
      "lighting_delta": l_delta,
      "active_bulbs": ctx.current_bulb_count,
      "expected_bulbs": ctx.baseline_bulb_count,
      "out_of_view_bulbs": getattr(ctx, "out_of_view_bulbs", []),
      "missing_items": ctx.missing_items,
      "drift_alerts": ctx.drift_alerts,
      "clutter_items": ctx.clutter_items,
      "checklist": ctx.reset_checklist,
      "annotated_image_path": out_path,
  }