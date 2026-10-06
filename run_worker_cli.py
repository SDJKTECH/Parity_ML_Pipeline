"""Worker Daily Capture Audit CLI (GUI File Picker)

Audits the worker capture against the baseline and prints a clear summary.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
for p in (REPO_ROOT, PIPELINE_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

from Pipeline.config import BASELINES_DIR, CURRENT_DIR, OUTPUT_DIR
from Pipeline.entrypoints.inference_service import evaluate_worker_capture


def open_file_dialog(initial_dir: str) -> str | None:
  root = tk.Tk()
  root.withdraw()
  root.attributes("-topmost", True)
  start_dir = initial_dir if os.path.isdir(initial_dir) else os.getcwd()

  print("\n[INFO] Opening File Explorer... Select the worker's capture image.")
  file_path = filedialog.askopenfilename(
      title="Select Worker Capture Image to Audit",
      initialdir=start_dir,
      filetypes=[
          ("Image Files", "*.jpg *.jpeg *.png *.webp"),
          ("All Files", "*.*"),
      ],
  )
  root.destroy()
  return file_path if file_path else None


def main():
  print("\n" + "=" * 60)
  print(" 👷 WORKER DAILY ROOM INSPECTION AUDIT")
  print("=" * 60)

  if not os.path.isdir(BASELINES_DIR):
    print(f"[ERROR] Baseline directory does not exist: {BASELINES_DIR}")
    return

  registered_rooms = [
      fname[:-len("_baseline.json")]
      for fname in sorted(os.listdir(BASELINES_DIR))
      if fname.endswith("_baseline.json")
  ]

  if not registered_rooms:
    print(f"[ERROR] No registered baselines found in '{BASELINES_DIR}'.")
    return

  selected_capture_path = open_file_dialog(CURRENT_DIR)
  if not selected_capture_path:
    print("\n[CANCELLED] No capture image selected.")
    return

  filename = os.path.basename(selected_capture_path)
  guessed_room = os.path.splitext(filename)[0]

  chosen_room = None
  if guessed_room in registered_rooms:
    confirm = (
        input(f"Audit against '{guessed_room}' baseline? (Y/n): ")
        .strip()
        .lower()
    )
    if not confirm or confirm == "y":
      chosen_room = guessed_room

  if not chosen_room:
    print("\nRegistered Baselines Available:")
    for idx, r in enumerate(registered_rooms, start=1):
      print(f"  [{idx}] {r}")
    choice = input(
        f"\nSelect room baseline [1-{len(registered_rooms)}]: "
    ).strip()
    if not (choice.isdigit() and 1 <= int(choice) <= len(registered_rooms)):
      print("[ERROR] Invalid selection.")
      return
    chosen_room = registered_rooms[int(choice) - 1]

  print("\n" + "=" * 60)
  print(f" ⚙️ RUNNING AUDIT: {chosen_room.upper()} ({filename})")
  print("=" * 60)

  report = evaluate_worker_capture(
      current_image_input=selected_capture_path,
      room_name=chosen_room,
      baseline_dir=BASELINES_DIR,
      output_dir=OUTPUT_DIR,
  )

  # Display Clean Executive Report Card
  print("\n" + "=" * 60)
  print(f" 📋 AUDIT REPORT: {chosen_room.upper()}")
  print("=" * 60)

  mode = report.get("audit_mode", "ALL")
  is_ok = report.get("verdict") == "OK"
  verdict_badge = "✅ PASSED" if is_ok else "❌ REJECTED"

  print(f"  Audit Mode     : {mode} INSPECTION")
  print(f"  Verdict        : {verdict_badge}")
  print(f"  Alignment SSIM : {report.get('ssim_score', 0):.2f}")

  if mode in ("BULBS", "ALL"):
    print(
        f"  Active Bulbs   : {report.get('active_bulbs', 0)} /"
        f" {report.get('expected_bulbs', 0)} Expected"
    )
    if report.get("out_of_view_bulbs"):
      print(
          f"  Camera Angle   : {len(report['out_of_view_bulbs'])} fixture(s)"
          " outside field of view"
      )

  print("\n  Audit Findings:")
  checklist = report.get("checklist", [])
  issues = [
      item
      for item in checklist
      if not item.startswith("[OK]") and not item.startswith("[INFO]")
  ]

  if not issues:
    print("    • All verification points matched baseline standards.")
  else:
    for issue in issues:
      print(f"    • {issue}")

  annotated_img = report.get("annotated_image_path")
  if annotated_img and os.path.exists(annotated_img):
    print(f"\n  Annotated Image: {annotated_img}")

  print("=" * 60 + "\n")


if __name__ == "__main__":
  main()