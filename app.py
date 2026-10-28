"""Streamlit web application for Night-to-Day conversion and YOLO object detection."""

from __future__ import annotations

import io
import tempfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Night-to-Day AI Detection",
    page_icon="☀️",
    layout="wide",
)


@st.cache_resource
def load_yolo_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Loads and caches the YOLO model in memory."""
    return YOLO(model_name)


def enhance_to_daylight(image_bgr: np.ndarray) -> np.ndarray:
    """Universal low-light to daylight enhancement pipeline.

    Applies LAB-space CLAHE, adaptive gamma correction based on mean luminance,
    and HSV saturation boosting to simulate daylight vibrancy.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)

    mean_lum = float(np.mean(l_clahe))
    # Darker images get stronger gamma lifting (smaller gamma exponent < 1.0 lifts shadows)
    gamma = 0.45 if mean_lum < 80 else (0.65 if mean_lum < 140 else 0.85)
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    l_bright = cv2.LUT(l_clahe, table)

    enhanced_bgr = cv2.cvtColor(cv2.merge((l_bright, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    h_chan, s_chan, v_chan = cv2.split(hsv)
    s_chan = np.clip(s_chan * 1.25, 0, 255)
    v_chan = np.clip(v_chan * 1.05, 0, 255)
    return cv2.cvtColor(cv2.merge((h_chan, s_chan, v_chan)).astype("uint8"), cv2.COLOR_HSV2BGR)


# Sidebar UI
st.sidebar.title("Configuration")
mode = st.sidebar.radio(
    "Select Input Mode", ["Multiple Images (Batch)", "Video File"]
)
model_choice = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Select the YOLO model architecture. yolov8n is fastest for real-time processing.",
)
conf_thresh = st.sidebar.slider(
    "Detection Confidence Threshold", 0.10, 0.90, 0.30, 0.05
)

st.sidebar.info(
    "🔒 **Privacy Notice**: All processing happens strictly in RAM. "
    "No uploaded or processed files are permanently stored on disk."
)

st.title("☀️ Night-to-Day Conversion & Object Detection")
st.caption(
    "Transform low-light, nighttime images and videos into clear daylight equivalents, "
    "then detect objects in real time using YOLOv8."
)

model = load_yolo_model(model_choice)

if mode == "Multiple Images (Batch)":
    uploaded_files = st.file_uploader(
        "Upload one or more night images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"Loaded **{len(uploaded_files)}** image(s) for processing.")

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            # Read image entirely in-memory (RAM)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if orig_bgr is None:
                st.warning(f"Could not decode image: {uploaded_file.name}")
                continue

            # Enhance & Detect
            daylight_bgr = enhance_to_daylight(orig_bgr)
            results = model.predict(source=daylight_bgr, conf=conf_thresh, verbose=False)
            annotated_bgr = results[0].plot()

            # Combine side-by-side
            combined_bgr = np.hstack((orig_bgr, annotated_bgr))
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

            num_detected = len(results[0].boxes)
            st.subheader(
                f"{idx}. {uploaded_file.name} — {num_detected} Object(s) Detected"
            )
            st.image(
                combined_rgb,
                caption="Left: Original Night Input | Right: Daylight + YOLO Detection",
                use_container_width=True,
            )

            # In-memory download buffer
            pil_img = Image.fromarray(combined_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button(
                label=f"📥 Download Combined Image ({uploaded_file.name})",
                data=buf.getvalue(),
                file_name=f"combined_{uploaded_file.name}.png",
                mime="image/png",
                key=f"dl_{idx}",
            )
            st.divider()

else:
    uploaded_video = st.file_uploader("Upload a night video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.info("Video uploaded. Initializing in-memory frame pipeline...")

        # Process using auto-deleting temporary files
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_in, \
             tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_out:

            temp_in.write(uploaded_video.read())
            temp_in.flush()

            cap = cv2.VideoCapture(temp_in.name)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (w * 2, h))
            progress_bar = st.progress(0, text="Processing video frames in memory...")

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                enhanced = enhance_to_daylight(frame)
                res = model.predict(source=enhanced, conf=conf_thresh, verbose=False)
                annotated = res[0].plot()
                out.write(np.hstack((frame, annotated)))

                frame_idx += 1
                if total_frames > 0:
                    pct = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(
                        pct, text=f"Processing frame {frame_idx}/{total_frames} ({int(pct * 100)}%)"
                    )

            cap.release()
            out.release()

            with open(temp_out.name, "rb") as f:
                video_bytes = f.read()

            st.success("✅ Video processing complete! Temporary files cleaned up.")
            st.download_button(
                "📥 Download Combined Video",
                data=video_bytes,
                file_name="daylight_detection_output.mp4",
                mime="video/mp4",
            )
