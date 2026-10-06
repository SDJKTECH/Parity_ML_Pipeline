"""Owner Master Registration CLI (Bulk GUI File Picker with Custom Prompt Setup)"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
for p in (REPO_ROOT, PIPELINE_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

from Pipeline.config import BASELINES_DIR, MASTER_DIR, TESTING_DIR
from Pipeline.entrypoints.master_service import get_engines, process_master_image


def open_bulk_file_dialog(initial_dir: str) -> list[str]:
  root = tk.Tk()
  root.withdraw()
  root.attributes("-topmost", True)
  start_dir = initial_dir if os.path.isdir(initial_dir) else os.getcwd()
  print("[INFO] Opening File Explorer... Select master baseline images.")
  file_paths = filedialog.askopenfilenames(
      title="Select Master Baseline Images",
      initialdir=start_dir,
      filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")],
  )
  root.destroy()
  return list(file_paths) if file_paths else []


def main():
  print("\n" + "=" * 60)
  print(" 🏢 OWNER MASTER BASELINE REGISTRATION")
  print("=" * 60)

  selected_image_paths = open_bulk_file_dialog(MASTER_DIR)
  if not selected_image_paths:
    print("\n[CANCELLED] No files selected.")
    return

  # Configure Prompt (mirrors the frontend modal)
  print("\n--- Configure Inspection Prompt ---")
  print("Presets:")
  print("  [1] Ceiling Lighting & Bulb Inspection")
  print("  [2] Staff Area Cleanliness & Furniture Drift Check")
  print("  [3] Custom Prompt Instructions")

  choice = input("Select prompt preset [1-3] (Default: 2): ").strip()
  if choice == "1":
    prompt_name = "Ceiling Lighting Inspection"
    prompt_instructions = (
        "Check if all ceiling tube lights and hanging bulbs are turned on."
    )
  elif choice == "3":
    prompt_name = (
        input("Enter Prompt Name: ").strip() or "Custom Room Inspection"
    )
    prompt_instructions = (
        input("Enter Prompt Instructions: ").strip()
        or "Inspect room state against baseline."
    )
  else:
    prompt_name = "Staff Area Cleanliness Check"
    prompt_instructions = (
        "Inspect chair alignment, check for missing plants, tables and clutter."
    )

  print(f"\n[CONFIG] Preset Name : {prompt_name}")
  print(f"[CONFIG] Instructions: {prompt_instructions}")
  print("[INFO] Preparing vision extractors...")
  get_engines()

  results = []
  print("\n" + "=" * 60)
  print(" ⚙️ REGISTERING BASELINES")
  print("=" * 60)

  for idx, img_path in enumerate(selected_image_paths, start=1):
    room_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[{idx}/{len(selected_image_paths)}] Processing '{room_name}'...")
    try:
      data = process_master_image(
          image_input=img_path,
          room_name=room_name,
          prompt_name=prompt_name,
          prompt_instructions=prompt_instructions,
          output_baseline_dir=BASELINES_DIR,
          export_annotated_dir=TESTING_DIR,
      )
      obj_count = len(data.get("objects", []))
      bulb_count = data.get("bulb_count", 0)
      results.append(
          (room_name, "SUCCESS", f"{obj_count} objects, {bulb_count} bulbs")
      )
    except Exception as exc:
      results.append((room_name, "FAILED", str(exc)))

  print("\n" + "=" * 60)
  print(" 📋 REGISTRATION SUMMARY")
  print("=" * 60)
  for room_name, status, details in results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"  {icon} {room_name:<20} : {status:<8} ({details})")
  print(f"\nBaselines ready in: {BASELINES_DIR}")
  print("=" * 60 + "\n")


if __name__ == "__main__":
  main()