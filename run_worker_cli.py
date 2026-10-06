"""
Worker Daily Capture Audit CLI (GUI File Picker)
Opens a native file explorer to pick the daily capture and audits against
the registered baseline stored in Pipeline/baselines/.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog

# Ensure repository root and Pipeline package are on sys.path
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
for p in (REPO_ROOT, PIPELINE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from Pipeline.config import CURRENT_DIR, BASELINES_DIR, OUTPUT_DIR
from Pipeline.entrypoints.inference_service import evaluate_worker_capture


def open_file_dialog(initial_dir: str) -> str | None:
    """Opens a native Windows/OS file explorer brought to front."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    start_dir = initial_dir if os.path.isdir(initial_dir) else os.getcwd()

    print("\n[INFO] Opening File Explorer window... Select the worker's capture image.")
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

    # 1. Discover registered baselines strictly in Pipeline/baselines
    if not os.path.isdir(BASELINES_DIR):
        print(f"[ERROR] Baseline directory does not exist: {BASELINES_DIR}")
        print("Please run 'python Pipeline/run_master_cli.py' first.")
        return

    registered_rooms = []
    for fname in sorted(os.listdir(BASELINES_DIR)):
        if fname.endswith("_baseline.json"):
            room = fname[:-len("_baseline.json")]
            registered_rooms.append(room)

    if not registered_rooms:
        print(f"[ERROR] No registered baselines found in '{BASELINES_DIR}'.")
        print("Please run 'python Pipeline/run_master_cli.py' first to register room baselines.")
        return

    # 2. Pick the capture image via native file dialog
    selected_capture_path = open_file_dialog(CURRENT_DIR)
    if not selected_capture_path:
        print("\n[CANCELLED] No capture image selected. Exiting.")
        return

    filename = os.path.basename(selected_capture_path)
    guessed_room = os.path.splitext(filename)[0]

    print(f"\n[SELECTED CAPTURE] {selected_capture_path}")

    # 3. Match with registered room
    chosen_room = None
    if guessed_room in registered_rooms:
        print(f" -> Automatically matched room baseline: '{guessed_room}'")
        confirm = input(f"Audit against '{guessed_room}' baseline? (Y/n): ").strip().lower()
        if not confirm or confirm == "y":
            chosen_room = guessed_room

    if not chosen_room:
        print("\nRegistered Baselines Available:")
        for idx, r in enumerate(registered_rooms, start=1):
            print(f"  [{idx}] {r}")

        choice = input(f"\nSelect room baseline [1-{len(registered_rooms)}]: ").strip()
        if not (choice.isdigit() and 1 <= int(choice) <= len(registered_rooms)):
            print("[ERROR] Invalid room selection. Exiting.")
            return
        chosen_room = registered_rooms[int(choice) - 1]

    # 4. Run the audit
    print("\n" + "=" * 60)
    print(f" ⚙️ RUNNING AUDIT: {chosen_room.upper()} ({filename})")
    print("=" * 60)

    report = evaluate_worker_capture(
        current_image_input=selected_capture_path,
        room_name=chosen_room,
        baseline_dir=BASELINES_DIR,
        output_dir=OUTPUT_DIR,
    )

    # 5. Display comprehensive audit report
    print("\n" + "=" * 60)
    print(f" 📋 AUDIT REPORT FOR: {chosen_room.upper()}")
    print("=" * 60)
    verdict_icon = "✅" if report.get("verdict") == "OK" else "❌"
    mode = report.get("audit_mode", "ALL")
    print(f"  • Mode           : {mode}")
    print(f"  • Verdict        : {verdict_icon} {report.get('verdict')}")
    print(f"  • SSIM Score     : {report.get('ssim_score', 0):.2f}")
    print(
        f"  • Lighting Delta : {report.get('lighting_delta', 0):+.2f} intensity"
        " units"
    )

    if mode in ("BULBS", "ALL"):
      print(
          f"  • Active Bulbs   : {report.get('active_bulbs', 0)} (Expected"
          f" Baseline: {report.get('expected_bulbs', 0)})"
      )

    print("\n  • Reset Checklist:")
    for item in report.get("checklist", []):
      print(f"     {item}")

    annotated_img = report.get("annotated_image_path")
    if annotated_img and os.path.exists(annotated_img):
      print(f"\n  • Annotated Visual Saved: {annotated_img}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()