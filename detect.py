"""Universal low-light to daylight enhancement and YOLO object detection CLI."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def enhance_to_daylight(image: np.ndarray) -> np.ndarray:
    """Universal low-light to daylight enhancement pipeline that works on any image.

    Combines denoising, LAB-space CLAHE, adaptive gamma correction, and saturation boost.
    """
    # 1. Mild denoising to prevent amplifying sensor noise in dark regions
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )

    # 2. Convert to LAB color space to isolate luminance (L) from color (A, B)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on Luminance
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # 4. Adaptive Gamma Correction based on mean image brightness
    mean_luminance = float(np.mean(l_clahe))
    # Darker images get stronger gamma lifting (smaller gamma exponent < 1.0 lifts shadows)
    gamma = 0.45 if mean_luminance < 80 else (0.65 if mean_luminance < 140 else 0.85)
    lookup_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    l_brightened = cv2.LUT(l_clahe, lookup_table)

    # Merge LAB channels back to BGR
    enhanced_lab = cv2.merge((l_brightened, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 5. Boost color saturation slightly in HSV space to simulate daylight vibrancy
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    h_channel, s_channel, v_channel = cv2.split(hsv)
    s_channel = np.clip(s_channel * 1.25, 0, 255)
    v_channel = np.clip(v_channel * 1.05, 0, 255)
    enhanced_hsv = cv2.merge((h_channel, s_channel, v_channel)).astype("uint8")

    daylight_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return daylight_img


def create_side_by_side(
    original_img: np.ndarray, detected_img: np.ndarray, filename: str = ""
) -> np.ndarray:
    """Combines original and processed images side-by-side with top header labels."""
    h, w, _ = original_img.shape

    # Add top banner for clean titles
    banner_height = max(40, int(h * 0.08))
    font_scale = max(0.6, w / 900.0)
    thickness = max(1, int(font_scale * 2))

    orig_padded = cv2.copyMakeBorder(
        original_img, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )
    det_padded = cv2.copyMakeBorder(
        detected_img, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(240, 240, 240)
    )

    cv2.putText(
        orig_padded,
        f"Original Input ({filename})",
        (15, int(banner_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        det_padded,
        "Daylight Conversion + Object Detection",
        (15, int(banner_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )

    # Add a thin divider line between images
    divider = np.full((orig_padded.shape[0], 4, 3), (180, 180, 180), dtype=np.uint8)
    combined = np.hstack((orig_padded, divider, det_padded))
    return combined


def process_all_images(
    input_source: str,
    conf_threshold: float = 0.25,
    save_outputs: bool = False,
    output_dir: str = "outputs",
    no_display: bool = False,
) -> None:
    """Processes a single image or a folder of images.

    Set save_outputs=False to enforce strict zero-storage (in-memory view only).
    """
    # Load YOLOv8 model (automatically downloads weights on first run)
    model = YOLO("yolov8n.pt")

    # Collect all supported image paths
    supported_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff")
    image_paths = []

    if os.path.isdir(input_source):
        for ext in supported_extensions:
            image_paths.extend(glob.glob(os.path.join(input_source, ext)))
            image_paths.extend(glob.glob(os.path.join(input_source, ext.upper())))
        image_paths = sorted(list(set(image_paths)))
    elif os.path.isfile(input_source):
        image_paths = [input_source]
    else:
        print(f"[ERROR] Invalid input path: {input_source}")
        return

    if not image_paths:
        print(f"[WARNING] No valid images found in: {input_source}")
        return

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_paths)} image(s). Starting pipeline...")
    if not no_display:
        print("Controls: Press ANY KEY on the image window to advance to the next image, or 'q' to quit.\n")

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        original_img = cv2.imread(img_path)

        if original_img is None:
            print(f"[{idx}/{len(image_paths)}] Skipping corrupted/unreadable file: {filename}")
            continue

        # Step 1: Enhance low-light image to daylight equivalent
        daylight_img = enhance_to_daylight(original_img)

        # Step 2: Run YOLO Object Detection on the enhanced image
        results = model.predict(source=daylight_img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()  # Draws bounding boxes & class labels

        # Count detected objects
        num_objects = len(results[0].boxes)
        print(f"[{idx}/{len(image_paths)}] Processed '{filename}' -> Detected {num_objects} object(s).")

        # Step 3: Stitch Original and Output side-by-side
        combined_image = create_side_by_side(original_img, annotated_img, filename)

        # Optional saving (Disabled by default for Zero-Data-Storage policy)
        if save_outputs:
            out_path = os.path.join(output_dir, f"combined_{filename}")
            cv2.imwrite(out_path, combined_image)
            print(f"       -> Saved output to: {out_path}")

        # Step 4: Display scaled preview window in-memory (if display is enabled)
        if not no_display:
            display_h, display_w = combined_image.shape[:2]
            max_screen_width = 1400
            if display_w > max_screen_width:
                scale = max_screen_width / display_w
                preview = cv2.resize(
                    combined_image, (int(display_w * scale), int(display_h * scale))
                )
            else:
                preview = combined_image

            try:
                cv2.imshow("Batch Night-to-Day Object Detection (In-Memory)", preview)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print("Exiting batch processing early.")
                    break
            except cv2.error:
                print("Note: GUI display not available in headless environment. Skipping imshow.")
                no_display = True

    if not no_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print("Batch processing finished! All in-memory buffers cleared.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Low-Light to Daylight Enhancement & YOLO Detection CLI"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for YOLO detection (default: 0.30).",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save combined comparison images to output directory (default: False for RAM-only).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output images if --save is enabled (default: 'outputs').",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive OpenCV imshow preview window.",
    )

    args = parser.parse_args()

    # If --input wasn't passed via CLI flags, prompt interactively
    input_source = args.input
    if not input_source:
        try:
            input_source = input(
                "Enter image file path OR folder path containing images: "
            ).strip().strip('"').strip("'")
        except EOFError:
            print("[ERROR] No input provided.")
            sys.exit(1)

    if not input_source:
        print("[ERROR] Input path cannot be empty.")
        sys.exit(1)

    process_all_images(
        input_source=input_source,
        conf_threshold=args.conf,
        save_outputs=args.save,
        output_dir=args.output_dir,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
