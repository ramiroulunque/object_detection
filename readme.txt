# 👁️ Real-Time Object Detection with OpenCV + YOLOv8

A Python project that uses your Mac's built-in camera to detect and recognize objects in real time using OpenCV and YOLOv8. Built as an intro to Computer Vision (CV), with the goal of eventually deploying to a Raspberry Pi or Arduino setup.

---

## Features

- Real-time object detection via webcam
- 80 detectable object classes out of the box (people, bottles, laptops, phones, cars, etc.)
- Color-coded bounding boxes with confidence scores
- Live FPS counter and object count overlay
- Zoom control (digital zoom in/out)
- Desktop notification + sound alert when a specific object is detected
- Screenshot capture
- Pause / resume

---

## Requirements

- Python 3.x
- Mac (tested on macOS with built-in camera)
- pip3

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ramiroulunque/object_detection.git
   cd object_detection
   ```

2. **Install dependencies**
   ```bash
   pip3 install opencv-python ultralytics
   ```

3. **Run the script**
   ```bash
   python3 object_detection.py
   ```

> On first run, YOLOv8 will automatically download the model (~6 MB). After that it runs instantly.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `P` | Pause / Resume |

---

## Configuration

All settings are at the top of `object_detection.py`:

```python
CAMERA_INDEX   = 0          # 0 = built-in Mac camera
CONFIDENCE     = 0.45       # Detection confidence threshold (0–1)
ZOOM           = 1.5        # > 1 zooms out, 1.0 = no change
ALERT_OBJECT   = "bottle"   # Object that triggers the alert
ALERT_COOLDOWN = 5          # Seconds between alerts
```

To watch for a different object, just change `ALERT_OBJECT` to any of the supported classes (e.g. `"person"`, `"cell phone"`, `"car"`, `"laptop"`).

---

## Supported Object Classes

YOLOv8 (COCO dataset) detects 80 classes including:

`person` · `bicycle` · `car` · `motorcycle` · `airplane` · `bus` · `train` · `truck` · `boat` · `traffic light` · `fire hydrant` · `stop sign` · `bench` · `bird` · `cat` · `dog` · `horse` · `backpack` · `umbrella` · `handbag` · `bottle` · `cup` · `fork` · `knife` · `spoon` · `bowl` · `banana` · `apple` · `pizza` · `laptop` · `mouse` · `keyboard` · `cell phone` · `chair` · `couch` · `bed` · `toilet` · `tv` · `microwave` · `oven` · `refrigerator` · `book` · `clock` · `vase` · `scissors` and more.

---

## Roadmap

- [ ] Log detections to CSV/JSON file
- [ ] Object count tracking over time
- [ ] Face blur / redaction mode
- [ ] Deploy to Raspberry Pi
- [ ] Trigger GPIO output on Arduino via serial when object detected

---

## Tech Stack

- [OpenCV](https://opencv.org/) — camera capture and image processing
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection model
- Python 3

