# Brighten + Detect

A Streamlit computer-vision app that:

1. Aggressively brightens dark images with OpenCV.
2. Detects common objects with a pretrained YOLO model.
3. Shows the original, enhanced, and annotated images side by side.
4. Provides a detection table and downloadable PNG outputs.

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
