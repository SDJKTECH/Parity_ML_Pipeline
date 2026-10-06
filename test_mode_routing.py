import os
import sys

# Add project root (parent directory of Pipeline/) to Python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
  if p not in sys.path:
    sys.path.insert(0, p)

# Now imports will work whether executed from Cafe/ or Pipeline/
from Pipeline.entrypoints.inference_service import evaluate_worker_capture

# Test 1: Bulb Prompt
print("\n" + "=" * 50)
print("TEST 1: Bulb Verification Prompt")
print("=" * 50)
res_bulbs = evaluate_worker_capture(
    current_image_input="current_images/FirstFloor.jpg",
    room_name="FirstFloor",
    prompt_text=(
        "Check if all ceiling tube lights and hanging bulbs are turned on."
    ),
)
print(f"Mode Selected : {res_bulbs['audit_mode']}")
print(f"Checklist     : {res_bulbs['checklist']}")
print(f"Verdict       : {res_bulbs['verdict']}")

# Test 2: Object / Cleanliness Prompt
print("\n" + "=" * 50)
print("TEST 2: Furniture Drift & Cleanliness Prompt")
print("=" * 50)
res_objects = evaluate_worker_capture(
    current_image_input="current_images/FirstFloor.jpg",
    room_name="FirstFloor",
    prompt_text="Inspect chair alignment, check for missing plants and clutter.",
)
print(f"Mode Selected : {res_objects['audit_mode']}")
print(f"Checklist     : {res_objects['checklist']}")
print(f"Verdict       : {res_objects['verdict']}")