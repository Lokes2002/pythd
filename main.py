from fastapi import FastAPI, Request, HTTPException
from PIL import Image
import pytesseract
import io

app = FastAPI(title="OCR + AI Suggestions API")

# Prompt-based suggestion system
def generate_suggestions(text: str):
    text_lower = text.lower()
    tips = []

    if len(text.strip()) < 10:
        tips.append("Add more meaningful content to engage users.")
    if "follow" in text_lower or "like" in text_lower:
        tips.append("Avoid asking for likes or follows directly; use value-based CTA.")
    if len(text.split()) > 150:
        tips.append("Make the content shorter and more readable.")
    if "motivation" in text_lower:
        tips.append("Add a personal story to increase emotional engagement.")
    if "business" in text_lower or "startup" in text_lower:
        tips.append("Use a hook line to capture attention early.")
    if "friend" in text_lower or "selfie" in text_lower:
        tips.append("Use well-lit images to improve post quality.")
    if not tips:
        tips.append("Try adding a strong hook in the first 2 lines for high engagement.")
    
    return tips


@app.get("/")
async def home():
    return {"status": "running", "info": "upload image to /extract"}


@app.post("/extract")
async def extract(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="No image received")

    try:
        img = Image.open(io.BytesIO(data))
    except:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # OCR
    text = pytesseract.image_to_string(img).strip()

    # Suggestions based on extracted text
    suggestions = generate_suggestions(text)

    return {
        "text": text,
        "suggestions": suggestions
    }
