"""Streamlit app for aggressive image brightening and object detection."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="Brighten + Detect",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load and cache the YOLO model so it is not reloaded per interaction."""
    return YOLO(model_name)


def brighten_image(
    image: np.ndarray,
    brightness: int,
    contrast: float,
    gamma: float,
    color_boost: float,
) -> np.ndarray:
    """Brighten an image using LAB illumination, contrast, gamma, and saturation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    lightness = cv2.convertScaleAbs(lightness, alpha=contrast, beta=brightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )

    gamma_table = np.array(
        [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    enhanced = cv2.LUT(enhanced, gamma_table)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * color_boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a BGR OpenCV image as downloadable PNG bytes."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("The processed image could not be encoded as PNG.")
    return encoded.tobytes()


def detection_rows(result) -> list[dict[str, object]]:
    """Convert YOLO detections into rows suitable for a Streamlit table."""
    names = result.names
    rows = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        rows.append(
            {
                "Object": names[class_id],
                "Confidence": f"{float(box.conf[0].item()) * 100:.1f}%",
                "X1": int(box.xyxy[0][0].item()),
                "Y1": int(box.xyxy[0][1].item()),
                "X2": int(box.xyxy[0][2].item()),
                "Y2": int(box.xyxy[0][3].item()),
            }
        )
    return rows


st.title("🔎 Brighten + Detect")
st.caption("Aggressively enhance dark images with OpenCV, then detect objects with YOLO.")

with st.sidebar:
    st.header("Enhancement controls")
    brightness = st.slider("Brightness", 0, 150, 65)
    contrast = st.slider("Contrast", 0.8, 2.5, 1.25, 0.05)
    gamma = st.slider("Gamma lift", 0.4, 2.5, 1.8, 0.1)
    color_boost = st.slider("Color boost", 0.8, 2.0, 1.15, 0.05)

    st.header("Detection controls")
    model_name = st.selectbox("YOLO model", ("yolo11n.pt", "yolo11s.pt"))
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.35, 0.05)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use a JPG or PNG image.",
)

if uploaded_file is None:
    st.info("Upload an image to begin. The first detection may download the selected YOLO model.")
    st.stop()

try:
    source_rgb = np.array(Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB"))
except (OSError, ValueError) as error:
    st.error(f"Unable to read this image: {error}")
    st.stop()

source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
bright_bgr = brighten_image(source_bgr, brightness, contrast, gamma, color_boost)

with st.spinner("Running object detection..."):
    model = load_model(model_name)
    result = model.predict(bright_bgr, conf=confidence, verbose=False)[0]
annotated_bgr = result.plot()
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
bright_rgb = cv2.cvtColor(bright_bgr, cv2.COLOR_BGR2RGB)

left, middle, right = st.columns(3)
with left:
    st.subheader("Original")
    st.image(source_rgb, use_container_width=True)
with middle:
    st.subheader("Brightened")
    st.image(bright_rgb, use_container_width=True)
with right:
    st.subheader("Detected objects")
    st.image(annotated_rgb, use_container_width=True)

rows = detection_rows(result)
st.subheader(f"Detection results ({len(rows)} objects)")
if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.warning("No objects were detected. Try lowering the confidence threshold.")

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Download brightened image",
        data=image_to_png_bytes(bright_bgr),
        file_name="brightened.png",
        mime="image/png",
        use_container_width=True,
    )
with download_col2:
    st.download_button(
        "Download detection result",
        data=image_to_png_bytes(annotated_bgr),
        file_name="brightened_detected.png",
        mime="image/png",
        use_container_width=True,
    )
