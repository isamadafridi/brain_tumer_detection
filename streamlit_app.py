import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

MODEL_PATH = "best.pt"

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("Brain Tumor Detection (Demo)")
st.warning(
    "⚠️ Research/learning demo only. Trained on a small 2-class dataset "
    "(negative/positive). Not validated for clinical use — do not use "
    "this for real diagnostic decisions."
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

COLORS = {"negative": "red", "positive": "blue"}

# conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
sensitivity = st.slider("Detection sensitivity", 0.0, 1.0, 0.75, 0.05)
conf_threshold = round(1.0 - sensitivity, 2)  # high sensitivity = low conf threshold
st.caption(f"Confidence threshold: {conf_threshold} — slide right to detect more, left to detect less")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Original")
        st.image(image, width=350)

    if st.button("Run detection"):
        with st.spinner("Running inference..."):
            results = model.predict(source=image, conf=conf_threshold)
            result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls)
            detections.append({
                "class": result.names[cls_id],
                "confidence": round(float(box.conf), 4),
                "box_xyxy": box.xyxy[0].tolist(),
            })

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for det in detections:
            x1, y1, x2, y2 = det["box_xyxy"]
            color = COLORS.get(det["class"], "yellow")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 12)), f"{det['class']} {det['confidence']:.0%}", fill=color)

        with col2:
            st.subheader("Detections")
            st.image(annotated, width=350)

        with col3:
            st.subheader("Detection details")
            if not detections:
                st.info("No detections at this confidence threshold.")
            else:
                for det in detections:
                    st.write(f"- **{det['class']}** -- {det['confidence']:.1%} confidence")
