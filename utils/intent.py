"""Intent Classifier to route pipeline execution based on the Owner's prompt."""

from __future__ import annotations
import re

BULB_KEYWORDS = {
    "bulb",
    "bulbs",
    "light",
    "lights",
    "lighting",
    "lamp",
    "lamps",
    "fixture",
    "fixtures",
    "chandelier",
    "illumination",
    "luminaire",
    "glow",
    "dim",
    "dark",
    "bright",
    "brightness",
}

OBJECT_KEYWORDS = {
    "chair",
    "chairs",
    "table",
    "tables",
    "furniture",
    "plant",
    "plants",
    "potted",
    "clutter",
    "clean",
    "cleanliness",
    "tidy",
    "tidiness",
    "drift",
    "missing",
    "refrigerator",
    "tv",
    "sofa",
    "couch",
    "seating",
    "arrangement",
    "desk",
    "desks",
    "person",
    "shoes",
}


def classify_prompt_mode(prompt_text: str | None) -> str:
  """Determines whether to run the Bulb pipeline or the Object pipeline.

  Returns:
      'BULBS' or 'OBJECTS' (defaults to 'OBJECTS' if ambiguous or empty)
  """
  if not prompt_text or not prompt_text.strip():
    return "OBJECTS"

  text = prompt_text.lower()
  words = set(re.findall(r"\b\w+\b", text))

  bulb_matches = words.intersection(BULB_KEYWORDS)
  object_matches = words.intersection(OBJECT_KEYWORDS)

  if len(bulb_matches) > len(object_matches):
    return "BULBS"
  return "OBJECTS"