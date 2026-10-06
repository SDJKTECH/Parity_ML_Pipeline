"""Sequential stages for the inspection pipeline."""

from . import (
    step1_alignment,
    step2_feature_extraction,
    step3_ssim_lighting,
    step3b_bulb_detection,
    step4_delta_checklist,
    step5_vlm_inspection,
)

__all__ = [
    "step1_alignment",
    "step2_feature_extraction",
    "step3_ssim_lighting",
    "step3b_bulb_detection",
    "step4_delta_checklist",
    "step5_vlm_inspection",
]