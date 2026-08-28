# Smoke Detection System — Detect Smoke Before Fire Becomes Obvious

A Python-based computer vision project that detects **smoke** in video footage using a custom-trained **YOLOv8** model and **OpenCV**. Smoke is often the earliest visible warning sign of a fire — appearing before flames are large enough to see clearly, or even before flames are visible at all. This system processes video frame-by-frame, draws bounding boxes around detected smoke, overlays a warning label, and saves the annotated result as a new video file.

---

## Why Smoke Detection?

Traditional fire detection reacts once flames are already visible — by then, the situation may already be escalating. Smoke, however, frequently appears **minutes earlier**, especially in slow-developing fires (electrical faults, smoldering materials, enclosed spaces). Detecting smoke early gives more time to respond, evacuate, or intervene before a fire becomes severe.

---

## Features

- Loads a custom-trained YOLOv8 smoke detection model (`smoke.pt`)
- Automatically detects input video resolution and FPS — no manual configuration needed
- Draws real-time bounding boxes around detected smoke regions
- Overlays an amber **"SMOKE DETECTED"** label with confidence score
- Saves the fully processed video to `output_video.mp4`
- Live preview window with a `q` key shortcut to stop processing early
- Beginner-friendly code with clear inline comments

---

## Tech Stack

- [Python 3.9+](https://www.python.org/)
- [OpenCV](https://opencv.org/) — video I/O, drawing, and display
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection framework
- [PyTorch](https://pytorch.org/) — backend inference engine for YOLO

---

## Project Structure

```
SmokeDetection/
│
├── smoke.pt              # Custom-trained YOLOv8 smoke detection model
├── input_video.mp4        # Input video to run detection on
├── main.py                 # Main detection script
├── output_video.mp4       # Generated output (created after running the script)
└── README.md               # Project documentation
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/SmokeDetection.git
   cd SmokeDetection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python
   ```

4. **Add your files**
   - Place your trained model file as `smoke.pt` in the project root.
   - Place your test footage as `input_video.mp4` in the project root.

---

## Usage

Run the script from the project directory:

```bash
python main.py
```

- A live preview window will show detections as the video processes.
- Press **`q`** at any time to stop processing early.
- Once finished (or stopped), the annotated video is saved as `output_video.mp4` in the project folder.

---

## How It Works

1. The script loads the custom YOLOv8 model (`smoke.pt`) using the Ultralytics API.
2. It reads the input video's width, height, and FPS so the output video matches exactly.
3. Each frame is passed through the YOLO model for inference.
4. Detected smoke regions are drawn as amber bounding boxes with a confidence-scored label.
5. Every processed frame — whether smoke is detected or not — is written to the output video file.
6. Resources (video capture, video writer, display windows) are properly released at the end.

---

## Example Output

| Input Frame | Detected Output |
|-------------|------------------|
| Raw video frame | Frame with amber bounding box + "SMOKE DETECTED" label |

*(Add a sample screenshot or GIF here once you have one — it makes the README much more compelling.)*

---

## Limitations & Notes

- Detection accuracy depends entirely on the quality and diversity of the dataset the model was trained on.
- Fog, mist, steam, clouds, and dust can visually resemble smoke and may occasionally trigger false positives depending on training data.
- Thin or translucent smoke in early stages can be harder to detect than dense smoke — consider a lower confidence threshold if early detection is critical, at the cost of more false positives.
- This project is intended for research, learning, and prototyping purposes — not as a certified life-safety system.
- For production/safety-critical smoke monitoring, use in conjunction with proper smoke detectors, particulate sensors, and professional-grade fire alarm systems.

---

## Future Improvements

- [ ] Add fire as a secondary detection class (for combined smoke + fire alerts)
- [ ] Add email/SMS alert integration on detection
- [ ] Deploy as a real-time webcam/CCTV monitoring tool
- [ ] Add a Streamlit/Gradio web interface for easier use
- [ ] Log detection events with timestamps to a CSV file
- [ ] Add confidence-threshold tuning for earlier (but noisier) smoke warnings

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework
- The open-source computer vision community for smoke/fire detection datasets and research
