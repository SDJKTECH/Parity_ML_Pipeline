from __future__ import annotations
from typing import Any, Callable
from Pipeline.context import PipelineContext
from Pipeline.utils.intent import classify_prompt_mode

# Import stages directly to prevent circular imports:
from Pipeline.stages.step1_alignment import run as run_step1
from Pipeline.stages.step2_feature_extraction import run as run_step2
from Pipeline.stages.step3_ssim_lighting import run as run_step3
from Pipeline.stages.step3b_bulb_detection import run as run_step3b
from Pipeline.stages.step4_delta_checklist import run as run_step4
from Pipeline.stages.step5_vlm_inspection import run as run_step5

StepCallable = Callable[[PipelineContext], bool]


class PipelineRunner:
  """Pluggable, prompt-routed pipeline execution engine."""

  def __init__(self, steps: list[Any] | None = None):
    self.explicit_steps = steps

  def _get_step_label(self, step_fn: StepCallable) -> str:
    module_name = getattr(step_fn, "__module__", "")

    name_map = {
        "step1_alignment": "Image Alignment & Homography",
        "step2_feature_extraction": "Object & Feature Extraction",
        "step3_ssim_lighting": "Lighting & SSIM Comparison",
        "step3b_bulb_detection": "Bulb & Fixture Verification",
        "step4_delta_checklist": "Delta & Inventory Checklist",
        "step5_vlm_inspection": "Multimodal VLM Visual Inspection",
    }

    for key, display_name in name_map.items():
      if key in module_name:
        return display_name

    return (
        getattr(step_fn, "__name__", "Processing Step")
        .replace("_", " ")
        .title()
    )

  def _resolve_steps_for_mode(self, mode: str) -> list[StepCallable]:
    """Dynamically builds stage sequence so only ONE pipeline branch runs."""
    if mode == "BULBS":
      # BULB PIPELINE: Skip YOLO Object Extraction (step 2)
      return [
          run_step1,
          run_step3,
          run_step3b,
          run_step4,
          run_step5,
      ]
    elif mode == "OBJECTS":
      # OBJECT PIPELINE: Skip Bulb Detector & CLIP Classifier (step 3b)
      return [
          run_step1,
          run_step2,
          run_step3,
          run_step4,
          run_step5,
      ]
    else:
      # ALL / Fallback: Run both
      return [
          run_step1,
          run_step2,
          run_step3,
          run_step3b,
          run_step4,
          run_step5,
      ]

  def run(self, ctx: PipelineContext) -> PipelineContext:
    """Classifies prompt intent and runs only the relevant branch."""
    # 1. Resolve execution mode if not manually preset
    current_mode = getattr(ctx, "audit_mode", "ALL")
    if current_mode == "ALL":
      prompt = (
          getattr(ctx, "prompt_text", None)
          or getattr(ctx, "room_rule", None)
          or ctx.master_data.get("prompt_instructions")
          or ctx.master_data.get("prompt_name")
      )
      ctx.audit_mode = classify_prompt_mode(prompt)
    else:
      ctx.audit_mode = current_mode

    print(
        f"\n[ORCHESTRATOR] Prompt-Routed Mode: >>> {ctx.audit_mode} <<<"
        f" ({'Bulb Verification Only' if ctx.audit_mode == 'BULBS' else 'Object / Drift Audit Only'})"
    )

    # 2. Select stage sequence
    steps_to_run = (
        self.explicit_steps
        if self.explicit_steps is not None
        else self._resolve_steps_for_mode(ctx.audit_mode)
    )

    # 3. Execute stages sequentially
    for step_fn in steps_to_run:
      step_name = self._get_step_label(step_fn)

      if getattr(ctx, "halt", False):
        print(f"\n[PIPELINE ABORTED] Inspection stopped early.")
        break

      success = step_fn(ctx)

      if not success or getattr(ctx, "halt", False):
        ctx.halt = True
        print("\n" + "=" * 55)
        print(f"[PIPELINE STOPPED] Verification failed at: {step_name}")
        errors = getattr(ctx, "errors", [])
        if errors:
          print(f"Reason: {errors[-1]}")
        print("Downstream verification skipped.")
        print("=" * 55 + "\n")
        break

    return ctx