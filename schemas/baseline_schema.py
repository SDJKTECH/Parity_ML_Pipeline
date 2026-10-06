from __future__ import annotations
from typing import Any, List
from pydantic import BaseModel, Field


class BulbCoordinate(BaseModel):
    bbox: List[int]  # [x1, y1, x2, y2]
    area: float = 0.0
    confidence: float = 1.0


class MasterBaselineSchema(BaseModel):
    room_name: str
    bulb_count: int = 0
    bulbs: List[BulbCoordinate] = Field(default_factory=list)
    furniture_detections: List[dict[str, Any]] = Field(default_factory=list)
    keypoint_count: int = 0