from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image

app = Flask(__name__)

# === Path ke model ===
model_path = 'saved_model_palm_disease.keras'

# Unduh model dari Google Drive jika belum ada
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1g-QPUIsySVm1oBl0KXpKKlxe7x_JPe7B'
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Label klasifikasi sesuai urutan output model
labels = ['Boron Excess', 'Ganoderma', 'Healthy', 'Scale insect']

@app.route('/')
def index():
    return "✅ API Deteksi Daun Sawit Siap Digunakan!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Gambar tidak ditemukan!'}), 400

    file = request.files['image']
    
    try:
        # Proses gambar
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({
            'class': labels[class_index],
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Untuk deploy ke Railway (harus pakai PORT dari environment)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
