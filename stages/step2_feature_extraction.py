from __future__ import annotations
from Pipeline.context import PipelineContext
from Pipeline.utils.features import extract_features


def run(ctx: PipelineContext) -> bool:
    """STEP 2: Feature Extraction on Aligned Image (Pure In-Memory)."""
    if ctx.aligned_current_img is None:
        ctx.halt = True
        return False

    # Run feature extraction directly on the in-memory aligned image
    ctx.current_data, _ = extract_features(ctx.aligned_current_img, ctx.model)

    return True