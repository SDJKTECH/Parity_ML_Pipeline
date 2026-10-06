from __future__ import annotations
from typing import Any, Callable
from Pipeline.context import PipelineContext
from Pipeline.utils.intent import classify_prompt_mode

# Import stage functions directly to prevent circular imports[cite: 37]:
from Pipeline.stages.step1_alignment import run as run_step1
from Pipeline.stages.step2_feature_extraction import run as run_step2
from Pipeline.stages.step3_ssim_lighting import run as run_step3
from Pipeline.stages.step3b_bulb_detection import run as run_step3b
from Pipeline.stages.step4_delta_checklist import run as run_step4
from Pipeline.stages.step5_vlm_inspection import run as run_step5

StepCallable = Callable[[PipelineContext], bool]


class PipelineRunner:
  """Pluggable, prompt-routed pipeline execution engine with clean progress reporting[cite: 37]."""

  def __init__(self, steps: list[Any] | None = None):
    self.explicit_steps = steps

  def _get_user_friendly_label(self, step_fn: StepCallable) -> str:
    """Returns clean, non-technical stage progress descriptions."""
    fn_name = getattr(step_fn, "__name__", "")
    mod_name = getattr(step_fn, "__module__", "")

    if "step1_alignment" in mod_name or "step1" in fn_name:
      return "Perspective Alignment & Geometry Verification"
    if "step2_feature_extraction" in mod_name or "step2" in fn_name:
      return "Inventory & Furniture Detection (YOLO)"
    if "step3_ssim_lighting" in mod_name or "step3" in fn_name:
      return "Structural Similarity & Luminance Delta Analysis"
    if "step3b_bulb_detection" in mod_name or "step3b" in fn_name:
      return "Ceiling Light Candidate Extraction & Verification"
    if "step4_delta_checklist" in mod_name or "step4" in fn_name:
      return "Discrepancy Audit & Checklist Formulation"
    if "step5_vlm_inspection" in mod_name or "step5" in fn_name:
      return "Multimodal Semantic Visual Inspection (VLM)"

    return (
        getattr(step_fn, "__name__", "Processing Step")
        .replace("_", " ")
        .title()
    )

  def _resolve_steps_for_mode(self, mode: str) -> list[StepCallable]:
    """Dynamically builds stage sequence so only ONE pipeline branch executes[cite: 37]."""
    if mode == "BULBS":
      # BULBS MODE: Skip YOLO Object Extraction (Step 2)[cite: 37]
      return [
          run_step1,
          run_step3,
          run_step3b,
          run_step4,
          run_step5,
      ]
    elif mode == "OBJECTS":
      # OBJECTS MODE: Skip Bulb Detector & Candidate Crops (Step 3b)[cite: 37]
      return [
          run_step1,
          run_step2,
          run_step3,
          run_step4,
          run_step5,
      ]
    else:
      # FALLBACK: Run all stages[cite: 37]
      return [
          run_step1,
          run_step2,
          run_step3,
          run_step3b,
          run_step4,
          run_step5,
      ]

  def run(self, ctx: PipelineContext) -> PipelineContext:
    """Classifies prompt intent, selects steps, and prints clean progress logs[cite: 37]."""
    # 1. Resolve execution mode if not already set[cite: 37]
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

    mode_description = (
        "Lighting & Bulb Verification Only"
        if ctx.audit_mode == "BULBS"
        else "Furniture Alignment & Cleanliness Only"
    )

    print(f"\n[ORCHESTRATOR] Prompt Mode : >>> {ctx.audit_mode} <<<")
    print(f"[ORCHESTRATOR] Task Focus  : {mode_description}")

    # 2. Select stage sequence[cite: 37]
    steps_to_run = (
        self.explicit_steps
        if self.explicit_steps is not None
        else self._resolve_steps_for_mode(ctx.audit_mode)
    )

    total_steps = len(steps_to_run)

    # 3. Execute stages with clean progress indicators[cite: 37]
    for idx, step_fn in enumerate(steps_to_run, start=1):
      step_label = self._get_user_friendly_label(step_fn)

      if getattr(ctx, "halt", False):
        print(f"\n[PIPELINE ABORTED] Inspection stopped early.")
        break

      print(f"[{idx}/{total_steps}] {step_label}...", end=" ", flush=True)

      try:
        success = step_fn(ctx)
      except Exception as exc:
        success = False
        if hasattr(ctx, "errors"):
          ctx.errors.append(str(exc))

      if success and not getattr(ctx, "halt", False):
        print("Done.")
      else:
        ctx.halt = True
        print("FAILED.")
        print("\n" + "=" * 55)
        print(f"[PIPELINE STOPPED] Verification failed at: {step_label}")
        errors = getattr(ctx, "errors", [])
        if errors:
          print(f"Reason: {errors[-1]}")
        print("Downstream verification skipped.")
        print("=" * 55 + "\n")
        break

    return ctx