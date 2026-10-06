from __future__ import annotations
import os
from Pipeline.config import RULES_DIR

# Default search directory for rules
RULES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image_rules")
if not os.path.exists(RULES_DIR):
    RULES_DIR = "image_rules"

def get_room_rule(room_name: str) -> str | None:
    """Finds and loads room-specific guidelines from image_rules directory."""
    if not room_name or not os.path.exists(RULES_DIR):
        return None

    candidates = [
        f"{room_name}.txt",
        f"{room_name.lower()}.txt",
    ]
    for candidate in candidates:
        rule_path = os.path.join(RULES_DIR, candidate)
        if os.path.exists(rule_path):
            with open(rule_path, "r", encoding="utf-8") as f:
                return f.read().strip()

    if os.path.exists(RULES_DIR):
        for fname in os.listdir(RULES_DIR):
            if fname.lower() == f"{room_name.lower()}.txt":
                rule_path = os.path.join(RULES_DIR, fname)
                with open(rule_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
    return None