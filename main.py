from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import fitz           # PyMuPDF  (PDF Read)
import cv2
import numpy as np
import io

app = FastAPI(title="OCR + AI Suggestions API")

# 🔥 Allow CORS for React / Java
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✨ AI Suggestions from text
def generate_suggestions(text: str):
    text_low = text.lower()
    tips = []

    if len(text.strip()) < 20:
        tips.append("Content is short. Add more details to improve engagement.")
    if "like" in text_low or "follow" in text_low:
        tips.append("Instead of asking for likes or follows, try speaking value first.")
    if len(text.split()) > 120:
        tips.append("Content is too long. Make it short and readable.")
    if "motivation" in text_low:
        tips.append("Motivational content performs best with a real-life story.")
    if not tips:
        tips.append("Add a hook in the first 2 lines to capture attention.")

    return tips


# 🔥 OCR Engine (Image → Text)
def run_ocr_from_image(pil_img):
    # Convert to OpenCV
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)

    # AI-level preprocessing
    img = cv2.resize(img, None, fx=1.6, fy=1.6)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    # OCR
    return pytesseract.image_to_string(img, lang="eng").strip()


# 📌 Main API
@app.post("/extract")
async def extract(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="No file received")

    text = ""

    # 🔍 Detect PDF vs Image
    is_pdf = data[0:4] == b"%PDF"

    try:
        if is_pdf:
            # Read PDF
            pdf = fitz.open(stream=data, filetype="pdf")
            output = []
            for page in pdf:
                pix = page.get_pixmap(dpi=300)
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = run_ocr_from_image(pil_img)
                output.append(page_text)
            text = "\n".join(output).strip()

        else:
            # Direct image OCR
            pil_img = Image.open(io.BytesIO(data))
            text = run_ocr_from_image(pil_img)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Suggestions
    suggestions = generate_suggestions(text)

    return {
        "text": text,
        "suggestions": suggestions
    }


@app.get("/")
async def home():
    return {"status": "running", "message": "Upload to POST /extract"}
