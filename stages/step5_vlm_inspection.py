from __future__ import annotations
import os
import cv2
from Pipeline.utils.vlm import analyze_room_with_vlm, load_prompt
from Pipeline.utils.rules import get_room_rule
from Pipeline.context import PipelineContext
from Pipeline.config import BASELINES_DIR, MASTER_PROMPT_PATH


def run(ctx: PipelineContext) -> bool:
    """STEP 5: VLM Multimodal Visual Inspection."""
    print("\n--- VLM Visual Inspection ---")

    # 1. Load system prompt template
    if os.path.exists(MASTER_PROMPT_PATH):
        base_prompt = load_prompt(MASTER_PROMPT_PATH)
    else:
        base_prompt = (
            "Compare the CURRENT IMAGE against the baseline state established by "
            "the MASTER IMAGE and MASTER JSON."
        )

    # 2. Attach room-specific rules
    ctx.room_rule = get_room_rule(ctx.room_name)
    if ctx.room_rule:
        print(f"[INFO] Applied Room Rule for '{ctx.room_name}': {ctx.room_rule}")
        combined_prompt = (
            f"{base_prompt}\n\n### SPECIFIC ROOM RULES FOR {ctx.room_name.upper()}\n{ctx.room_rule}"
        )
    else:
        combined_prompt = base_prompt

    # 3. Check aligned frame availability
    target_img = ctx.aligned_current_img if ctx.aligned_current_img is not None else ctx.raw_current_img
    if target_img is None:
        print("[VLM ERROR] No image available for VLM inspection.")
        return False

    ref_img_path = os.path.join(BASELINES_DIR, f"{ctx.room_name}_ref.jpg")
    if not os.path.exists(ref_img_path):
        print(f"[VLM ERROR] Baseline reference image not found: {ref_img_path}")
        return False

    # 4. Encode image and send to Gemini
    success, encoded_img = cv2.imencode(".jpg", target_img)
    if not success:
        print("[VLM ERROR] Failed to encode current frame for VLM.")
        return False

    try:
        ctx.vlm_result = analyze_room_with_vlm(
            processed_image=encoded_img.tobytes(),
            master_image=ref_img_path,
            master_json=ctx.master_data,
            master_prompt=combined_prompt,
        )

        # 5. Restore full detailed logging
        print(f"[VLM STATUS] {ctx.vlm_result.status}")
        print(f"[VLM SUMMARY] {ctx.vlm_result.summary}")

        if ctx.vlm_result.issues:
            print("[VLM DETECTED DISCREPANCIES]")
            for issue in ctx.vlm_result.issues:
                print(
                    f" - [{issue.type}] ({issue.severity}) {issue.object}: "
                    f"{issue.description} (Confidence: {issue.confidence:.2f})"
                )
        else:
            print("[VLM] No visual discrepancies detected.")

    except Exception as e:
        print(f"[VLM ERROR] Inspection failed: {e}")

    return True