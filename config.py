"""
Pipeline Central Configuration & Path Resolver
"""
import os

# Base directory of the Pipeline package
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PIPELINE_DIR)

def find_input_dir(name: str) -> str:
    """Finds input source folders across common project locations."""
    candidates = [
        os.path.join(PIPELINE_DIR, name),
        os.path.join(os.getcwd(), name),
        os.path.join(PARENT_DIR, name),
        os.path.join(PARENT_DIR, "Nudge", name),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return os.path.join(PIPELINE_DIR, name)

# Input image directories (searched dynamically)
MASTER_DIR = find_input_dir("master_images")
CURRENT_DIR = find_input_dir("current_images")
RULES_DIR = find_input_dir("image_rules")

# Generated output directories (strictly inside Pipeline/)
BASELINES_DIR = os.path.join(PIPELINE_DIR, "baselines")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output_images")
TESTING_DIR = os.path.join(PIPELINE_DIR, "Testing")
PROMPTS_DIR = os.path.join(PIPELINE_DIR, "prompts")
MASTER_PROMPT_PATH = os.path.join(PROMPTS_DIR, "master_prompt.txt")

# Ensure required Pipeline directories exist
os.makedirs(BASELINES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TESTING_DIR, exist_ok=True)

# File Format Support
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Visual & Alignment Thresholds
MAX_DRIFT_PIXELS = 50.0
MIN_SSIM_SCORE = 0.80

# Bulb Detection Settings
DEFAULT_GAMMA = 2.2
DEFAULT_BULB_PAD = 45