# Universal Low-Light to Daylight Enhancement & YOLOv8 Object Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)](https://opencv.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.58+-red.svg)](https://streamlit.io/)
[![Zero Storage Guarantee](https://img.shields.io/badge/Privacy-Zero--Storage-brightgreen.svg)](#zero-data-storage--privacy-policy)

An end-to-end computer vision and AI pipeline designed to transform dark, underexposed, or nighttime images and videos into clear, daylight-equivalent scenes and execute real-time object detection with YOLOv8.

---

## 🌟 Key Features

- **5-Stage Low-Light to Daylight Enhancement Pipeline**:
  - **Fast Non-Local Means Denoising**: Suppresses high-frequency sensor noise in dark regions before brightening.
  - **LAB Color Space Decoupling**: Separates the luminance channel ($L^*$) from chromatic opponent channels ($a^*, b^*$).
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast equalization preventing over-amplification in already lit areas.
  - **Adaptive Non-Linear Gamma Correction**: Automatically computes mean scene luminance and applies an inverse gamma lookup table ($\gamma = 0.45, 0.65, 0.85$) scaled to the darkness level.
  - **HSV Color Saturation & Vibrancy Boost**: Simulates natural daylight color rendition by boosting saturation and value channels.
- **YOLOv8 Object Detection**:
  - Automatically identifies persons, vehicles, animals, and common everyday objects in formerly invisible shadows.
  - Displays bounding boxes with confidence scores.
- **Interactive Streamlit Web Application (`app.py`)**:
  - **Batch Images Mode**: Upload multiple nighttime photos simultaneously, process side-by-side in real-time, and download individual enhanced results.
  - **Video Processing Mode**: Process full video files frame-by-frame with a live progress indicator and download the stitched daylight comparison video.
- **Standalone CLI Pipeline (`detect.py`)**:
  - Supports single image path or whole directory batch processing.
  - Interactive file prompt or standard command-line flags (`--input`, `--conf`, `--save`, `--no-display`).
  - Resized in-memory visualizer with side-by-side banner comparisons.
- **Zero-Data-Storage Policy**:
  - Enforces in-memory (RAM) processing by default; no uploaded media is saved to disk unless explicitly requested via CLI flag `--save` or web download button.

---

## 🔬 How the Enhancement Pipeline Works

```
Raw Night Input
       │
       ▼
[1. Mild Denoising] ─────────► Suppress dark-region sensor noise (fastNlMeans)
       │
       ▼
[2. LAB Color Space] ────────► Isolate Luminance (L) from Color (A, B)
       │
       ▼
[3. CLAHE Processing] ───────► Adaptive Local Histogram Equalization (clipLimit=3.5)
       │
       ▼
[4. Adaptive Gamma LUT] ─────► Non-linear luminance lift based on mean scene brightness
       │
       ▼
[5. HSV Saturation Boost] ───► Enrich color vibrancy to mimic natural daylight
       │
       ▼
Enhanced Daylight Image
       │
       ▼
[YOLOv8 Inference] ──────────► Multi-class object detection on illuminated frame
       │
       ▼
Side-by-Side Comparison (Original vs. Daylight + Detections)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/samba18071-bot/opencv-bright-object-detector.git
cd opencv-bright-object-detector
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Option A: Interactive Streamlit Web App

Launch the local web application:

```bash
streamlit run app.py
```

1. Select **Multiple Images (Batch)** or **Video File** in the sidebar.
2. Adjust the **Detection Confidence Threshold** slider (default: `0.30`).
3. Select the YOLO model (`yolov8n.pt`, `yolov8s.pt`, or `yolov8m.pt`).
4. Upload your files:
   - For images: View the side-by-side comparison and click **Download Combined Image**.
   - For video: Monitor real-time frame progress and click **Download Combined Video**.

---

### Option B: Command Line Interface (`detect.py`)

Run interactively:

```bash
python detect.py
```
*You will be prompted to enter a file or directory path.*

Or pass command-line arguments directly:

```bash
# Process a single image (RAM-only preview)
python detect.py --input /path/to/dark_image.jpg

# Process a folder of images with custom confidence threshold
python detect.py --input /path/to/images/ --conf 0.35

# Save side-by-side comparison images to disk
python detect.py --input /path/to/images/ --save --output-dir outputs/

# Run headless without opening GUI windows
python detect.py --input /path/to/images/ --save --no-display
```

#### CLI Flags:
| Flag | Description | Default |
| --- | --- | --- |
| `-i, --input` | Path to image or directory containing images | Interactive prompt |
| `-c, --conf` | Confidence threshold for YOLO detection | `0.30` |
| `-s, --save` | Save combined comparison images to disk | `False` (in-memory only) |
| `-o, --output-dir` | Target directory when saving | `outputs/` |
| `--no-display` | Disable interactive OpenCV imshow preview window | `False` |

---

## 🔒 Zero-Data-Storage & Privacy Policy

- **Streamlit**: Images and video frames are decoded directly from memory buffers (`io.BytesIO`, `bytearray`). Video chunks use auto-deleting temporary files (`tempfile.NamedTemporaryFile(delete=True)`).
- **CLI**: By default, `save_outputs=False`. All processed arrays exist exclusively in RAM during runtime and are discarded when execution finishes.

---

## 📦 Project Structure

```
opencv-bright-object-detector/
├── app.py              # Streamlit interactive UI (Batch Images & Video)
├── detect.py           # CLI tool for batch / single image daylight detection
├── requirements.txt    # Production dependencies
├── .gitignore          # Ignored cache, model weights, and artifacts
└── README.md           # Documentation and guides
```

---

## 📄 License

This project is licensed under the MIT License - feel free to adapt and use for your computer vision and robotics projects!
