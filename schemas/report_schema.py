from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class InspectionReportSchema(BaseModel):
    room_name: str
    verdict: str  # "OK" or "REJECTED"
    ssim_score: float = 0.0
    lighting_delta: float = 0.0
    active_bulbs: int = 0
    expected_bulbs: int = 0
    missing_items: List[str] = Field(default_factory=list)
    drift_items: List[str] = Field(default_factory=list)
    clutter_items: List[str] = Field(default_factory=list)
    checklist: List[str] = Field(default_factory=list)
    vlm_feedback: Optional[str] = None
    annotated_image_bytes: Optional[bytes] = None