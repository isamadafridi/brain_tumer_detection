"""
FastAPI backend for brain tumor detection inference.
Loads best.pt once at startup, exposes POST /predict.
"""

import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Brain Tumor Detection API")

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=image, conf=conf)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        detections.append({
            "class": result.names[cls_id],
            "confidence": round(float(box.conf), 4),
            "box_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
        })

    return JSONResponse({
        "num_detections": len(detections),
        "detections": detections,
    })
