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
class_names = np.array(['metal', 'battery', 'plastic', 'shoes', 'paper', 'cardboard', 'glass', 'biological'])

# Inisialisasi FastAPI
app = FastAPI()

# === Tambahkan CORS Middleware ===
origins = [
    "http://localhost:3000",  # kalau React jalan di port 3000
    "http://localhost:5173",  # kalau pakai Vite
    "http://localhost:8097",  # sesuai port kamu
    "*",  # boleh akses dari mana saja (tidak disarankan di production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # atau ["*"] untuk semua
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Sampah Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Simpan file temporer
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Preprocessing gambar
        img = image.load_img(temp_path, target_size=(224, 224))  # sesuaikan dengan input model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]

        return {"predicted_class": pred_class, "confidence": float(np.max(prediction))}
    
    finally:
        os.remove(temp_path)