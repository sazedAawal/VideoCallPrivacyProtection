"""
Real-Time Person Blur / Block Tool
===================================
Tested on: macOS, Python 3.10+, VS Code

INSTALL (run once in your terminal):
    pip install ultralytics opencv-python numpy



CONTROLS (click inside the OpenCV window first):
    Left-click on a person  ->  cycle effect: none -> blur -> block -> none
    B                       ->  blur  ALL persons
    K                       ->  block ALL persons
    R                       ->  reset ALL effects
    Q or ESC                ->  quit
"""

import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "yolov8n.pt"  # downloads automatically (~6 MB on first run)
WEBCAM_INDEX  = 0             # 0 = built-in Mac camera; try 1 if wrong device
CONF_THRESH   = 0.45
BLUR_STRENGTH = 51            # must be odd; higher = stronger blur
WINDOW_NAME   = "PersonGuard  |  click person = cycle effect  |  B=blur all  K=block all  R=reset  Q=quit"

# BGR colours for bounding-box outlines
C_NORMAL = (0,   200,  80)   # green  — unaffected
C_BLUR   = (255, 165,   0)   # orange — blurred
C_BLOCK  = (0,    60, 220)   # blue   — blocked

EFFECT_CYCLE = ["none", "blur", "block"]
# ──────────────────────────────────────────────────────────────────────────────


def apply_blur(frame, x1, y1, x2, y2):
    """Gaussian-blur the region inside the bounding box."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
    frame[y1:y2, x1:x2] = blurred


def apply_block(frame, x1, y1, x2, y2):
    """Solid dark rectangle — completely hides the person."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), thickness=-1)


def draw_label(frame, text, x1, y1, color):
    """Small filled label above the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + base), color, -1)
    cv2.putText(frame, text, (x1 + 3, label_y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def point_in_box(px, py, x1, y1, x2, y2):
    return x1 <= px <= x2 and y1 <= py <= y2


def main():
    print("[INFO] Loading YOLOv8n — first run will download the model (~6 MB)...")
    model = YOLO(MODEL_NAME)
    print("[INFO] Model ready. Opening webcam...")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # track_id -> "none" | "blur" | "block"
    effects    = defaultdict(lambda: "none")
    last_boxes = []   # shared between main loop and mouse callback

    # ── Mouse callback ─────────────────────────────────────────────────────────
    def on_mouse(event, px, py, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for (tid, x1, y1, x2, y2) in last_boxes:
                if point_in_box(px, py, x1, y1, x2, y2):
                    current     = effects[tid]
                    next_effect = EFFECT_CYCLE[(EFFECT_CYCLE.index(current) + 1) % len(EFFECT_CYCLE)]
                    effects[tid] = next_effect
                    print(f"  [CLICK] Person ID {tid}  ->  {next_effect.upper()}")
                    break

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    frame_count = 0
    start_time  = time.time()

    print("[INFO] Window open. Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame — retrying...")
            continue

        # ── YOLOv8 + ByteTrack ────────────────────────────────────────────────
        results = model.track(
            source=frame,
            persist=True,           # keeps tracker state across frames
            classes=[0],            # 0 = person only
            conf=CONF_THRESH,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        current_boxes = []
        if results and results[0].boxes.id is not None: # type: ignore
            for box, tid in zip(
                results[0].boxes.xyxy.cpu().numpy(), # type: ignore
                results[0].boxes.id.cpu().numpy().astype(int), # type: ignore
            ):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                current_boxes.append((int(tid), x1, y1, x2, y2))

        last_boxes[:] = current_boxes  # update for mouse hit-testing

        # ── Render effects + bounding boxes ───────────────────────────────────
        for (tid, x1, y1, x2, y2) in current_boxes:
            effect = effects[tid]

            if effect == "blur":
                apply_blur(frame, x1, y1, x2, y2)
                color = C_BLUR
                label = f"ID {tid} | BLUR"

            elif effect == "block":
                apply_block(frame, x1, y1, x2, y2)
                color = C_BLOCK
                label = f"ID {tid} | BLOCKED"

            else:
                color = C_NORMAL
                label = f"ID {tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)

        # ── HUD: FPS + person count ────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0
        hud     = f"FPS: {fps:.1f}   Persons detected: {len(current_boxes)}"

        # shadow + main text for readability on any background
        cv2.putText(frame, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20),   2, cv2.LINE_AA)
        cv2.putText(frame, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard controls ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):       # Q or ESC
            print("[INFO] Quit.")
            break

        elif key == ord("b"):
            for (tid, *_) in current_boxes:
                effects[tid] = "blur"
            print("  [KEY] Blurred ALL visible persons")

        elif key == ord("k"):
            for (tid, *_) in current_boxes:
                effects[tid] = "block"
            print("  [KEY] Blocked ALL visible persons")

        elif key == ord("r"):
            effects.clear()
            print("  [KEY] Reset ALL effects")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()