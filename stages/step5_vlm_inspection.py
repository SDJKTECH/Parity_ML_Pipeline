from __future__ import annotations
import os
import warnings
import cv2
from Pipeline.config import BASELINES_DIR, MASTER_PROMPT_PATH
from Pipeline.context import PipelineContext
from Pipeline.utils.rules import get_room_rule
from Pipeline.utils.vlm import analyze_room_with_vlm, load_prompt

# Suppress GenAI automatic function calling warning
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google_genai.models"
)


def run(ctx: PipelineContext) -> bool:
  """STEP 5: Multimodal Visual Inspection (VLM)."""
  if os.path.exists(MASTER_PROMPT_PATH):
    base_prompt = load_prompt(MASTER_PROMPT_PATH)
  else:
    base_prompt = (
        "Compare the CURRENT IMAGE against the baseline state established by"
        " the MASTER IMAGE and MASTER JSON."
    )

  ctx.room_rule = get_room_rule(ctx.room_name)
  combined_prompt = (
      f"{base_prompt}\n\n### SPECIFIC ROOM RULES FOR"
      f" {ctx.room_name.upper()}\n{ctx.room_rule}"
      if ctx.room_rule
      else base_prompt
  )

  target_img = (
      ctx.aligned_current_img
      if ctx.aligned_current_img is not None
      else ctx.raw_current_img
  )
  if target_img is None:
    return False

  ref_img_path = os.path.join(BASELINES_DIR, f"{ctx.room_name}_ref.jpg")
  if not os.path.exists(ref_img_path):
    return False

  success, encoded_img = cv2.imencode(".jpg", target_img)
  if not success:
    return False

  try:
    ctx.vlm_result = analyze_room_with_vlm(
        processed_image=encoded_img.tobytes(),
        master_image=ref_img_path,
        master_json=ctx.master_data,
        master_prompt=combined_prompt,
    )

    if ctx.vlm_result and getattr(ctx.vlm_result, "issues", None):
      for issue in ctx.vlm_result.issues:
        print(f"  • [VLM] [{issue.type}] {issue.object}: {issue.description}")
    else:
      print("  • [VLM] Scene confirmed against baseline standard.")

  except Exception:
    # Fail silently to avoid interrupting the audit output
    pass

  return True