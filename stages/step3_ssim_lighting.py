from __future__ import annotations
import cv2
from skimage.metrics import structural_similarity as ssim
from Pipeline.context import PipelineContext


def run(ctx: PipelineContext) -> bool:
    """STEP 3: SSIM & Lighting Delta Calculation."""
    if ctx.master_img is None or ctx.aligned_current_img is None:
        return False

    master_gray = cv2.cvtColor(
        cv2.resize(ctx.master_img, (640, 480)), cv2.COLOR_BGR2GRAY
    )
    aligned_gray = cv2.cvtColor(
        cv2.resize(ctx.aligned_current_img, (640, 480)), cv2.COLOR_BGR2GRAY
    )

    ctx.ssim_score = float(ssim(master_gray, aligned_gray))
    diff = float(
        ctx.current_data.get("brightness", 0.0)
        - ctx.master_data.get("brightness", 0.0)
    )

    ctx.brightness_diff = diff
    ctx.lighting_delta = diff  # Populates the key read by inference_service

    print(f"[INFO] Alignment SSIM Score: {ctx.ssim_score:.2f}")
    print(f"[INFO] Lighting Delta: {ctx.lighting_delta:+.2f} intensity units")
    return True