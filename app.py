from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import easyocr

app = FastAPI(
    title="YOLO Detection API",
    description="Upload image bytes to /detect to detect objects using YOLOv8",
    version="1.0.0"
)

# 👉 Ensure yolov8n.pt same folder me ho
model = YOLO("yolov8n.pt")


@app.get("/")
async def home():
    return {"message": "YOLO Detection API is running. Go to /docs for UI."}


@app.post("/detect")
async def detect(request: Request):
    # body se raw bytes lo
    data = await request.body()

    if not data:
        raise HTTPException(status_code=400, detail="No image data received")

    # bytes → PIL image
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # YOLO inference
    try:
        results = model(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detections.append({
            "label": label,
            "confidence": confidence
        })

    return {"objects": detections}



# Load OCR once
ocr_reader = easyocr.Reader(['en'])

@app.post("/ocr")
async def ocr(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="No image data received")

    try:
        result = ocr_reader.readtext(io.BytesIO(data), detail=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    return {"text": " ".join(result)}
