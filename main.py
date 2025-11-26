from fastapi import FastAPI, Request, HTTPException
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io

app = FastAPI(title="OCR + AI Suggestions API")

def generate_suggestions(text: str):
    tips = []
    t = text.lower()

    if len(text.strip()) < 10: tips.append("Add more content to increase engagement.")
    if "follow" in t or "like" in t: tips.append("Avoid asking directly for likes/follows.")
    if len(text.split()) > 100: tips.append("Make the content short and clean.")
    if not tips: tips.append("Add a hook in first 2 lines to catch attention.")
    return tips

@app.post("/extract")
async def extract(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="No image received")

    try:
        pil_img = Image.open(io.BytesIO(data))
    except:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # OCR preprocessing (magic part 🔥)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(img).strip()
    suggestions = generate_suggestions(text)

    return {"text": text, "suggestions": suggestions}
