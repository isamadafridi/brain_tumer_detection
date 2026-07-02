"""
Streamlit UI. Sends uploaded image to local FastAPI backend (running in
the same container on port 8000) and displays results side by side,
with detection details in a third column to the right.
"""

import streamlit as st
import requests
from PIL import Image, ImageDraw

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("Brain Tumor Detection (Demo)")
st.warning(
    "⚠️ Research/learning demo only. Trained on a small 2-class dataset "
    "(negative/positive). Not validated for clinical use — do not use "
    "this for real diagnostic decisions."
)

conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Original")
        st.image(image, width=350)

    if st.button("Run detection"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files, params={"conf": conf_threshold})

        if response.status_code != 200:
            st.error(f"API error: {response.status_code} - {response.text}")
        else:
            data = response.json()
            detections = data["detections"]

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for det in detections:
                x1, y1, x2, y2 = det["box_xyxy"]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 12)), f"{det['class']} {det['confidence']:.0%}", fill="red")

            with col2:
                st.subheader("Detections")
                st.image(annotated, width=350)

            with col3:
                st.subheader("Detection details")
                if data["num_detections"] == 0:
                    st.info("No detections at this confidence threshold.")
                else:
                    for det in detections:
                        st.write(f"- **{det['class']}** -- {det['confidence']:.1%} confidence")