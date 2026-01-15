import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS # <-- Wajib buat connect ke Frontend

app = Flask(__name__)
CORS(app) # <-- Izinkan semua akses (Biar frontend gak kena blokir)

# --- 1. LOAD MODEL ---
# Pastikan file ini ikut di-upload ke GitHub nanti!
MODEL_PATH = 'best_grapho_model.h5' 
model = tf.keras.models.load_model(MODEL_PATH)

LABELS = ['Tipe 1', 'Tipe 2', 'Tipe 3', 'Tipe 4', 'Tipe 5', 
          'Tipe 6', 'Tipe 7', 'Tipe 8', 'Tipe 9']

# --- 2. FUNGSI PREPROCESSING ---
def preprocess_image(image_stream):
    # Baca gambar dari memory
    file_bytes = np.frombuffer(image_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Resize ke 128x128 (Sesuai training kamu)
    img = cv2.resize(img, (128, 128))
    
    # --- METODE HE (Histogram Equalization) ---
    img = cv2.equalizeHist(img)
    img = cv2.bitwise_not(img) # Invert
    
    # Normalisasi & Reshape
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

@app.route('/', methods=['GET'])
def home():
    return "Server Grapholyze Aktif! (Histogram Equalization Ready)"

# --- 3. PINTU MASUK API ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400
    
    file = request.files['file']
    
    try:
        processed_img = preprocess_image(file.stream)
        
        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result_label = LABELS[class_idx]
        
        return jsonify({
            'status': 'success',
            'prediction': result_label,
            'confidence': f"{confidence*100:.2f}%",
            'message': 'Analisis Grafologi Selesai'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 4. SETTING PORT UNTUK RAILWAY ---
if __name__ == '__main__':
    # Railway akan kasih port lewat environment variable
    # Kalau di laptop lokal, dia pakai 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
