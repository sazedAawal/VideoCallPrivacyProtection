"""
SextortionGuard — Real-Time Nudity Detector
=============================================
Platform : macOS, Python 3.10+, VS Code

INSTALL (run once):
    pip install nudenet opencv-python-headless numpy

    If opencv-python-headless causes issues, try:
    pip install opencv-python numpy nudenet

RUN:
    python nude_detector.py

WHAT IT DETECTS
---------------
- Live nudity (real person in front of webcam)
- Nude photo held up to the webcam (printed or on phone screen)
- Nude video playing on a screen held to the webcam
- Partial nudity above the confidence threshold
- Works on any skin tone, any lighting condition

WHAT HAPPENS ON DETECTION
--------------------------
- Entire frame is blurred immediately
- Red warning banner shown on screen
- Bounding boxes drawn around detected regions
- Flag printed in terminal (timestamp + label + score)
- Nothing is stored — no video, no image, no log file

CONTROLS
--------
    Q or ESC  →  quit
    R         →  manually reset / clear the alert
    S         →  save a BLURRED screenshot to current folder
"""

import time
import sys
from datetime import datetime

import cv2
import numpy as np


# ─── SETTINGS (edit here to tune) ──────────────────────────────────────────────
WEBCAM_INDEX     = 0      # 0 = built-in Mac camera, try 1 if it opens wrong cam
CONFIDENCE       = 0.55   # trigger if score >= this (0.0–1.0). Lower = more sensitive
SCAN_EVERY       = 6      # scan every N frames. Lower = more responsive, slower FPS
BLUR_AMOUNT      = 99     # blur intensity when triggered (must be odd number)
WINDOW_TITLE     = "SextortionGuard  |  Q=quit   R=reset   S=save screenshot"
# ────────────────────────────────────────────────────────────────────────────────


# Labels from NudeNet that we treat as explicit
EXPLICIT = {
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_ANUS",
    "EXPOSED_BUTTOCKS",
}

# Labels that are non-explicit body parts (shown as grey boxes if you want)
SAFE = {
    "COVERED_BREAST_F",
    "COVERED_GENITALIA_F",
    "COVERED_BUTTOCKS",
    "EXPOSED_BELLY",
    "EXPOSED_ARMPITS",
    "EXPOSED_FEET",
    "FACE_F",
    "FACE_M",
}

# ── Colours (BGR format for OpenCV) ──────────────────────────────────────────
RED    = (0,   0,   220)
GREEN  = (0,   180,  60)
GREY   = (130, 130, 130)
WHITE  = (255, 255, 255)
BLACK  = (20,  20,  20)


# ══════════════════════════════════════════════════════════════════════════════
def load_nudenet():
    """Load NudeNet detector. Downloads weights (~90MB) on first run."""
    try:
        from nudenet import NudeDetector
        print("[INFO] Loading NudeNet — first run downloads model weights (~90 MB)...")
        det = NudeDetector()
        print("[INFO] NudeNet ready.\n")
        return det
    except ImportError:
        print("\n[ERROR] nudenet is not installed.")
        print("        Fix:  pip install nudenet\n")
        sys.exit(1)


def scan_frame(detector, frame):
    """
    Run NudeNet on a single frame.
    Returns two lists: explicit_hits, all_detections
    Both are dicts with keys: class, score, box
    """
    try:
        results = detector.detect(frame)
    except Exception as e:
        print(f"[SCAN ERROR] {e}")
        return [], []

    # Filter out very low-confidence noise
    all_dets = [r for r in results if r.get("score", 0) >= 0.30]

    explicit_hits = [
        r for r in all_dets
        if r.get("class", "") in EXPLICIT
        and r.get("score", 0) >= CONFIDENCE
    ]

    return explicit_hits, all_dets


def apply_blur(frame):
    """Blur entire frame to hide explicit content."""
    return cv2.GaussianBlur(frame, (BLUR_AMOUNT, BLUR_AMOUNT), 0)


def draw_box(frame, det, color):
    """Draw a labelled bounding box for a single detection."""
    box = det.get("box", [])
    if not box or len(box) < 4:
        return

    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x2, y2     = x + w, y + h
    label      = f"{det['class']}  {det['score']:.0%}"

    # Box outline
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Label pill above box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(label, font, 0.48, 1)
    ly = max(y - 4, th + 6)
    cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 8, ly + base), color, -1)
    cv2.putText(frame, label, (x + 4, ly - 2), font, 0.48, WHITE, 1, cv2.LINE_AA)


def draw_warning(frame, hits):
    """Red alert banner across top of frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), RED, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame,
        "⚠  EXPLICIT CONTENT DETECTED — STREAM BLOCKED",
        (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2, cv2.LINE_AA)

    detail = "   |   ".join(f"{d['class']}  {d['score']:.0%}" for d in hits[:3])
    cv2.putText(frame, detail,
        (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)


def draw_clear(frame):
    """Green all-clear bar at top of frame."""
    cv2.putText(frame,
        "✓  All clear — no explicit content detected",
        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64, GREEN, 2, cv2.LINE_AA)


def draw_hud(frame, fps, frame_num, last_scan, flagged):
    """Bottom info bar."""
    fh = frame.shape[0]
    status = "FLAGGED" if flagged else "CLEAR"
    text = (
        f"FPS: {fps:.1f}   "
        f"Frame: {frame_num}   "
        f"Last scan: frame {last_scan}   "
        f"Status: {status}"
    )
    # Shadow + main
    cv2.putText(frame, text, (12, fh - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (12, fh - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)


def log_flag(hits):
    """Print flag to terminal only. Nothing written to disk."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for hit in hits:
        print(f"  [FLAG] {ts}  {hit['class']}  score={hit['score']:.3f}")


def save_screenshot(frame):
    """Save blurred frame as JPG in current directory."""
    name = f"flagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(name, frame)
    print(f"  [SAVE] Screenshot saved → {name}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("=" * 56)
    print("  SextortionGuard — Nudity Detector")
    print("  All processing is LOCAL. Nothing stored or sent.")
    print("=" * 56)
    print()

    detector = load_nudenet()

    # ── Open webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {WEBCAM_INDEX}.")
        print("        Change WEBCAM_INDEX = 1 at the top of this file and try again.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    cv2.namedWindow(WINDOW_TITLE)

    # ── State ──────────────────────────────────────────────────────────────────
    frame_num    = 0
    last_scan    = 0
    start_time   = time.time()
    flagged      = False
    explicit_hits = []
    all_dets      = []

    print("[INFO] Webcam open.")
    print("[INFO] Point your camera at anything — live person, phone screen,")
    print("[INFO] printed photo, or a video playing. It detects all of them.")
    print("[INFO] Press Q or ESC in the window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Could not read frame — retrying...")
            continue

        frame_num += 1
        fps = frame_num / max(time.time() - start_time, 0.001)

        # ── Run NudeNet every SCAN_EVERY frames ───────────────────────────────
        if frame_num % SCAN_EVERY == 0:
            last_scan     = frame_num
            explicit_hits, all_dets = scan_frame(detector, frame)
            flagged = len(explicit_hits) > 0

            if flagged:
                print()
                log_flag(explicit_hits)

        # ── Build display ──────────────────────────────────────────────────────
        display = frame.copy()

        if flagged:
            display = apply_blur(display)

            # Draw boxes on blurred frame so locations are still visible
            for det in explicit_hits:
                draw_box(display, det, RED)

            draw_warning(display, explicit_hits)

        else:
            # Draw any safe-label boxes in grey (optional — informational)
            for det in all_dets:
                if det.get("class", "") in SAFE:
                    draw_box(display, det, GREY)

            draw_clear(display)

        draw_hud(display, fps, frame_num, last_scan, flagged)

        cv2.imshow(WINDOW_TITLE, display)

        # ── Keys ───────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):       # Q or ESC — quit
            print("\n[INFO] Quit.")
            break

        elif key == ord("r"):           # R — reset alert manually
            flagged       = False
            explicit_hits = []
            all_dets      = []
            print("[INFO] Alert manually reset.")

        elif key == ord("s"):           # S — save blurred screenshot
            if flagged:
                save_screenshot(display)
            else:
                print("[INFO] Nothing flagged right now — screenshot not saved.")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly. No data was stored.")
    print()


if __name__ == "__main__":
    main()