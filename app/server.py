from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
import os
import gdown

# Path model dan URL Google Drive
MODEL_PATH = "app/best_model_efficientnetb7.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1T6CLe2uu5wK4oQDe4G6Pr1LIZLeAhu7L"

# Unduh model jika belum ada
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# Daftar class
class_names = np.array([
    'metal', 'battery', 'plastic', 'shoes',
    'paper', 'cardboard', 'glass', 'biological'
])

# Inisialisasi FastAPI
app = FastAPI()

# ✅ Konfigurasi CORS untuk akses React lokal & production
origins = [
    "http://localhost:3000",  # React dev local
    "https://ml-model-sortify-production.up.railway.app",
    "http://localhost:8097",  # React production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Sampah Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]

        return {
            "predicted_class": pred_class,
            "confidence": float(np.max(prediction))
        }
    finally:
        os.remove(temp_path)
