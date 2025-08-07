from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import io

app = FastAPI()

# --- Model Loading ---
# Load the pre-trained ViT model and processor
# This will be done once when the application starts.
model_name = 'google/vit-base-patch16-224'
try:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    # If the model fails to load, the app can still run but /classify will fail.
    processor = None
    model = None

# Mount the 'static' directory to serve the index.html file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the main page
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not model or not processor:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    # Read image file
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess the image and predict
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    prediction = model.config.id2label[predicted_class_idx]

    return {"prediction": prediction}

# To run the app: uvicorn main:app --reload
