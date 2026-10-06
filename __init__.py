"""
Pipeline: Computer Vision & Multimodal Inspection Engine.

Exposes high-level service functions for:
1. register_master_room: Owner reference baseline registration.
2. evaluate_worker_capture: Worker capture delta & discrepancy audit.
"""
from Pipeline.context import PipelineContext
from Pipeline.runner import PipelineRunner
from Pipeline.entrypoints.master_service import process_master_image
from Pipeline.entrypoints.inference_service import evaluate_worker_capture

__all__ = [
    "PipelineContext",
    "PipelineRunner",
    "process_master_image",
    "evaluate_worker_capture",
]