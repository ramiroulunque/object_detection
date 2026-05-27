"""
Real-Time Object Detection using OpenCV + YOLOv8
--------------------------------------------------
Uses your Mac's built-in camera to detect and label objects in real time.
Includes zoom control, sound alert, and desktop notification when a
specific object is detected.

SETUP (run once in your terminal):
    pip3 install opencv-python ultralytics

USAGE:
    python3 object_detection.py

CONTROLS:
    Q  — quit
    S  — save a screenshot
    P  — pause / resume
"""

import cv2
import time
import subprocess
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0              # 0 = built-in Mac camera
CONFIDENCE     = 0.45           # Min confidence to show a detection (0–1)
MODEL_NAME     = "yolov8n.pt"   # nano model: fast & lightweight (auto-downloads)
WINDOW_TITLE   = "Object Detection  |  Q = quit  S = screenshot  P = pause"
ZOOM           = 1.0            # > 1 = zoom out (try 1.2 to 2.0)
ALERT_OBJECT   = "bottle"       # object that triggers the alert
ALERT_COOLDOWN = 5              # seconds between alerts (avoids spam)
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — one colour per class index
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
    (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
]

def get_color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


def draw_box(frame, box, label: str, conf: float, class_id: int):
    """Draw a bounding box + label on the frame."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = get_color(class_id)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text  = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

    # Label text
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, count: int, paused: bool, alert_active: bool):
    """Draw FPS + object count in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    status = "PAUSED" if paused else f"FPS: {fps:.1f}"
    cv2.putText(frame, status,              (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
    cv2.putText(frame, f"Objects: {count}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

    # Alert flash banner
    if alert_active:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, f"  {ALERT_OBJECT.upper()} DETECTED!",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def send_alert(label: str):
    """Trigger a Mac sound + desktop notification."""
    subprocess.run([
        "osascript", "-e",
        f'display notification "A {label} was detected!" with title "CV Alert" sound name "Funk"'
    ])


def main():
    print(f"Loading model '{MODEL_NAME}'...  (downloads ~6 MB on first run)")
    model  = YOLO(MODEL_NAME)
    names  = model.names

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_INDEX in config.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Running!  Watching for '{ALERT_OBJECT}'. Press Q to quit, S to screenshot, P to pause.")

    paused          = False
    frame_time      = time.time()
    fps             = 0.0
    frozen          = None
    last_alert_time = 0
    alert_flash     = 0

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ── Keyboard controls ───────────────────────────────────────────────
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                frozen = display.copy() if 'display' in dir() else None
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.jpg"
            if 'display' in dir():
                cv2.imwrite(fname, display)
                print(f"Saved -> {fname}")

        # ── Pause ───────────────────────────────────────────────────────────
        if paused:
            if frozen is not None:
                draw_hud(frozen, fps, 0, paused=True, alert_active=False)
                cv2.imshow(WINDOW_TITLE, frozen)
            continue

        # ── Capture ─────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — retrying...")
            continue

        # ── Zoom ────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        new_w, new_h = int(w / ZOOM), int(h / ZOOM)
        frame = frame[cy - new_h//2 : cy + new_h//2, cx - new_w//2 : cx + new_w//2]
        frame = cv2.resize(frame, (w, h))

        # ── Inference ───────────────────────────────────────────────────────
        results = model(frame, conf=CONFIDENCE, verbose=False)[0]

        # ── Draw detections ─────────────────────────────────────────────────
        display = frame.copy()
        count   = 0

        for box in results.boxes:
            conf     = float(box.conf[0])
            class_id = int(box.cls[0])
            label    = names[class_id]
            draw_box(display, box, label, conf, class_id)
            count   += 1

            # ── Alert ────────────────────────────────────────────────────────
            if label == ALERT_OBJECT:
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN:
                    send_alert(label)
                    last_alert_time = now
                    alert_flash     = now

        # ── FPS ─────────────────────────────────────────────────────────────
        now        = time.time()
        fps        = 1.0 / max(now - frame_time, 1e-6)
        frame_time = now

        # Flash alert banner for 2 seconds after detection
        alert_active = (time.time() - alert_flash) < 2

        draw_hud(display, fps, count, paused=False, alert_active=alert_active)
        cv2.imshow(WINDOW_TITLE, display)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()