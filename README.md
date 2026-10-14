# Brighten + Detect

A Streamlit computer-vision app that:

1. Aggressively brightens dark images with OpenCV.
2. Detects common objects with a pretrained YOLO model.
3. Shows the original, enhanced, and annotated images side by side.
4. Provides a detection table and downloadable PNG outputs.

## Train and measure accuracy

The repository includes a reproducible training script using Ultralytics'
public COCO8 sample dataset. COCO8 is intentionally tiny, so it is suitable for
verifying the workflow but is not a production-quality training set.

```bash
python train.py --epochs 10 --device cpu
```

The script downloads `coco8.yaml` and the YOLO weights on first use, trains the
detector, then evaluates it on the dataset's validation split. It prints:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95

Training artifacts and the best weights are written under
`runs/detect/coco8_training/`. Do not treat COCO8 metrics as representative of
real-world accuracy; train on a larger, domain-specific, correctly annotated
dataset for a meaningful production result.

### COCO8 training result

Using `yolo11n.pt`, 10 CPU epochs, image size 640, and the default COCO8
validation split, the completed run reported:

| Metric | Result |
| --- | ---: |
| Precision | 73.74% |
| Recall | 93.33% |
| mAP@0.50 | 94.22% |
| mAP@0.50:0.95 | 65.97% |

These values are specific to this run and the tiny COCO8 sample. They should
not be presented as expected accuracy on a user-provided dataset.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first detection downloads the selected YOLO weights from Ultralytics. The app
does not require an API key.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and push this project:

   ```bash
   git init
   git add .
   git commit -m "Build OpenCV brightening and object detection app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repository>.git
   git push -u origin main
   ```

2. Open [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub,
   and choose **Create app**.
3. Select your repository, branch (`main`), and main file (`app.py`).
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

The free Streamlit runtime may take a little longer on the first request because
it downloads the YOLO weights and initializes the model.

## How enhancement works

The pipeline combines LAB lightness enhancement with CLAHE, brightness and
contrast adjustment, gamma correction, and HSV saturation control. This gives
strong visibility improvements while retaining more detail than simply adding
to every RGB channel.
