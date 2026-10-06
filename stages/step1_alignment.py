from __future__ import annotations
import cv2
from Pipeline.context import PipelineContext
from Pipeline.utils.align_images import (
    align_images,
)  # or utils.align_images depending on your import path


def run(ctx: PipelineContext) -> bool:
  """STEP 1: Alignment & Homography Warp."""
  if ctx.master_img is None or ctx.raw_current_img is None:
    print("[FAIL] Missing master or raw current image in context.")
    ctx.halt = True
    return False

  try:
    aligned_img, matrix, valid_mask = align_images(
        master_img=ctx.master_img, daily_img=ctx.raw_current_img
    )

    ctx.aligned_current_img = aligned_img
    ctx.homography_matrix = matrix
    ctx.valid_mask = valid_mask

    print(
        "[PASS] Homography Alignment Successful (Image perspective rectified)."
    )
    return True

  except ValueError as e:
    print(f"[FAIL] Alignment failed ({e}). Halting pipeline.")
    if hasattr(ctx, "errors"):
      ctx.errors.append(str(e))
    ctx.halt = True
    return False