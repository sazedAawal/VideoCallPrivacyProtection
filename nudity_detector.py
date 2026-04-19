"""
Real-Time Nudity Detector
==========================
Platform : macOS, Python 3.10+, VS Code

INSTALL (run once in terminal):
    pip install nudenet opencv-python numpy


WHAT IT DOES
------------
- Opens your webcam in real time
- Runs NudeNet on every N frames (configurable)
- Detects nudity whether it is a real person OR a photo/image
  held up to the camera (NudeNet is image-based, not person-based)
- If confidence >= threshold:
    → blurs the entire frame
    → shows a red warning banner
    → draws labelled boxes around detected regions
    → prints a log line in the terminal (timestamp + label + score)
- If all clear → green status bar shown

CONTROLS (click the OpenCV window first):
    Q / ESC  →  quit
    S        →  save a flagged screenshot (blurred) to disk
    R        →  reset / clear the current trigger manually
"""

import time
import cv2
import numpy as np
from datetime import datetime


# ─── CONFIG  (edit these to tune behaviour) ────────────────────────────────────
WEBCAM_INDEX      = 0      # 0 = built-in Mac camera; try 1 if wrong device
CONF_THRESHOLD    = 0.60   # flag if any explicit label exceeds this (0.0 – 1.0)
CHECK_EVERY       = 8      # run NudeNet every N frames (lower = slower but more responsive)
BLUR_STRENGTH     = 99     # emergency blur kernel size — must be odd number
SHOW_SAFE_BOXES   = False  # set True to also draw boxes for non-explicit detections
SAVE_SCREENSHOTS  = True   # allow 'S' key to save blurred screenshot
WINDOW_NAME       = "NudityGuard  |  Q=quit  S=save screenshot  R=reset"

# NudeNet labels considered EXPLICIT — triggers the alert
EXPLICIT_LABELS = {
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_ANUS",
    "EXPOSED_BUTTOCKS",
}

# NudeNet labels considered SAFE (drawn in grey if SHOW_SAFE_BOXES=True)
SAFE_LABELS = {
    "EXPOSED_BELLY",
    "COVERED_BELLY",
    "COVERED_BREAST_F",
    "COVERED_BUTTOCKS",
    "COVERED_GENITALIA_F",
    "EXPOSED_ARMPITS",
    "EXPOSED_FEET",
    "FACE_F",
    "FACE_M",
}

# Colours (BGR)
C_EXPLICIT = (0,   0,   220)   # red    — explicit detection box
C_SAFE     = (120, 120, 120)   # grey   — safe detection box
C_CLEAR    = (0,   180,  60)   # green  — all-clear bar
C_WARN     = (0,   0,   200)   # red    — warning banner fill
C_WHITE    = (255, 255, 255)
# ────────────────────────────────────────────────────────────────────────────────


def load_detector():
    """Load NudeNet. Downloads model weights on first run (~90 MB)."""
    try:
        from nudenet import NudeDetector
        print("[INFO] Loading NudeNet model — first run downloads weights (~90 MB)...")
        detector = NudeDetector()
        print("[INFO] NudeNet ready.\n")
        return detector
    except ImportError:
        print("[ERROR] nudenet not installed.")
        print("        Run:  pip install nudenet")
        raise


def run_detection(detector, frame) -> list[dict]:
    """
    Run NudeNet on a single frame.
    Returns list of dicts: {label, score, box: [x, y, w, h]}
    Only returns detections above a minimum confidence to reduce noise.
    """
    try:
        results = detector.detect(frame)
        # Filter out very low confidence noise
        return [r for r in results if r.get("score", 0) >= 0.30]
    except Exception as e:
        print(f"[WARN] Detection error: {e}")
        return []


def is_explicit(detections: list[dict], threshold: float) -> tuple[bool, list[dict]]:
    """
    Returns (triggered, list_of_explicit_detections_above_threshold).
    """
    hits = [
        d for d in detections
        if d.get("class", "") in EXPLICIT_LABELS and d.get("score", 0) >= threshold
    ]
    return len(hits) > 0, hits


def draw_detection_boxes(frame, detections: list[dict], threshold: float):
    """Draw bounding boxes for all detections on the frame."""
    for det in detections:
        label  = det.get("class", "")
        score  = det.get("score", 0.0)
        box    = det.get("box", [])

        if not box or len(box) < 4:
            continue

        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x2, y2     = x + w, y + h

        explicit = label in EXPLICIT_LABELS and score >= threshold
        safe     = label in SAFE_LABELS

        if explicit:
            color = C_EXPLICIT
        elif safe and SHOW_SAFE_BOXES:
            color = C_SAFE
        else:
            continue   # skip if not explicit and safe boxes are hidden

        # Bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        # Label background + text
        tag       = f"{label}  {score:.0%}"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.48
        thickness = 1
        (tw, th), base = cv2.getTextSize(tag, font, scale, thickness)
        tag_y = max(y - 4, th + 4)
        cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 6, tag_y + base), color, -1)
        cv2.putText(frame, tag, (x + 3, tag_y - 2), font, scale, C_WHITE, thickness, cv2.LINE_AA)


def draw_warning_banner(frame, hits: list[dict]):
    """Red banner at the top of the frame when explicit content is detected."""
    h, w     = frame.shape[:2]
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), C_WARN, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    cv2.putText(frame,
                "EXPLICIT CONTENT DETECTED — FLAGGED",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 2, cv2.LINE_AA)

    # List first 3 hits in the sub-line
    summary = "   |   ".join(
        f"{d['class']}  {d['score']:.0%}" for d in hits[:3]
    )
    cv2.putText(frame, summary,
                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)


def draw_clear_bar(frame):
    """Green status bar at top when no explicit content detected."""
    cv2.putText(frame,
                "All clear — no explicit content detected",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_CLEAR, 2, cv2.LINE_AA)


def draw_hud(frame, fps: float, frame_idx: int, last_check: int, triggered: bool):
    """Bottom HUD showing FPS and detection status."""
    h = frame.shape[0]
    status = "FLAGGED" if triggered else "CLEAR"
    color  = (0, 0, 200) if triggered else (0, 180, 60)
    info   = f"FPS: {fps:.1f}   Frame: {frame_idx}   Last scan: {frame_idx - last_check} frames ago   Status: {status}"

    cv2.putText(frame, info, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20),   2, cv2.LINE_AA)
    cv2.putText(frame, info, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)


def blur_frame(frame: np.ndarray) -> np.ndarray:
    """Apply heavy Gaussian blur to obscure the entire frame."""
    return cv2.GaussianBlur(frame, (BLUR_STRENGTH, BLUR_STRENGTH), 0)


def save_screenshot(frame: np.ndarray):
    """Save the current (already blurred) frame with a timestamp filename."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flagged_{ts}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[SAVE] Screenshot saved → {filename}")


def log_detection(hits: list[dict]):
    """Print a structured log line to terminal. No content stored."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for hit in hits:
        print(f"[FLAG] {ts}  label={hit['class']}  score={hit['score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("  NudityGuard — Real-Time Webcam Detector")
    print("=" * 58)
    print()

    detector = load_detector()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {WEBCAM_INDEX}.")
        print("        Try changing WEBCAM_INDEX to 1 at the top of the file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    cv2.namedWindow(WINDOW_NAME)

    frame_idx    = 0
    last_check   = 0
    start_time   = time.time()

    # Persistent detection state — holds result until next scan
    triggered    = False
    explicit_hits: list[dict] = []
    all_detections: list[dict] = []

    print("[INFO] Window open. Show your webcam to the window.")
    print("[INFO] Try holding a photo or image up to the camera.")
    print("[INFO] Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame, retrying...")
            continue

        frame_idx += 1
        fps = frame_idx / max(time.time() - start_time, 0.001)

        # ── Run NudeNet every CHECK_EVERY frames ──────────────────────────────
        if frame_idx % CHECK_EVERY == 0:
            last_check      = frame_idx
            all_detections  = run_detection(detector, frame)
            triggered, explicit_hits = is_explicit(all_detections, CONF_THRESHOLD)

            if triggered:
                log_detection(explicit_hits)

        # ── Build display frame ───────────────────────────────────────────────
        display = frame.copy()

        if triggered:
            display = blur_frame(display)
            draw_detection_boxes(display, explicit_hits, CONF_THRESHOLD)
            draw_warning_banner(display, explicit_hits)
        else:
            draw_detection_boxes(display, all_detections, CONF_THRESHOLD)
            draw_clear_bar(display)

        draw_hud(display, fps, frame_idx, last_check, triggered)

        cv2.imshow(WINDOW_NAME, display)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):           # Q or ESC — quit
            print("[INFO] Quit.")
            break

        elif key == ord("s") and SAVE_SCREENSHOTS:  # S — save screenshot
            if triggered:
                save_screenshot(display)
            else:
                print("[INFO] Nothing flagged — screenshot not saved.")

        elif key == ord("r"):               # R — manual reset
            triggered      = False
            explicit_hits  = []
            all_detections = []
            print("[INFO] Manually reset — detection cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()