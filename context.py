"""Pipeline Context: In-memory state passed between all pipeline stages."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class PipelineContext:
  room_name: str
  raw_current_img: np.ndarray
  master_img: np.ndarray
  master_data: dict[str, Any]
  model: Any = None
  img_path: str = "in_memory"
  filename: str = ""

  # Prompt & Dynamic Execution Mode
  prompt_text: str | None = None  # Raw prompt from owner
  audit_mode: str = "ALL"  # "BULBS", "OBJECTS", or "ALL"

  # Control flow & error logging
  halt: bool = False
  errors: list[str] = field(default_factory=list)

  # Stage 1: Homography & Alignment
  aligned_current_img: np.ndarray | None = None
  homography_matrix: np.ndarray | None = None
  valid_mask: np.ndarray | None = None
  ssim_score: float = 0.0
  lighting_delta: float = 0.0
  brightness_diff: float = 0.0

  # Stage 2: Feature & Object Extraction
  current_data: dict[str, Any] = field(default_factory=dict)
  current_features: dict[str, Any] = field(default_factory=dict)

  # Stage 3b: Bulb Detection
  current_bulb_count: int = 0
  baseline_bulb_count: int = 0
  bulb_detections: list[dict[str, Any]] = field(default_factory=list)
  bulb_annotated_img: np.ndarray | None = None

  # Stage 4: Delta Checklist & Alerts
  out_of_view_bulbs: list[int] = field(default_factory=list)
  missing_items: list[str] = field(default_factory=list)
  drift_items: list[str] = field(default_factory=list)
  drift_alerts: list[str] = field(default_factory=list)
  clutter_items: list[str] = field(default_factory=list)
  reset_checklist: list[str] = field(default_factory=list)
  verdict: str = "OK"  # "OK" or "REJECTED"

  # Stage 5: VLM Inspection
  room_rule: str | None = None
  vlm_result: Any = None