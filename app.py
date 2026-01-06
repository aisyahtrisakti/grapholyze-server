# Install dulu: pip install flask tensorflow opencv-python-headless numpy

from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# --- 1. LOAD MODEL KAMU ---
# Pastikan file model .h5 atau .keras ada di folder yang sama
MODEL_PATH = 'best_grapho_model.h5' 
model = tf.keras.models.load_model(MODEL_PATH)

# Label Enneagram (Sesuaikan urutan training!)
# Pastikan urutannya SAMA PERSIS dengan saat kamu training (le.classes_)
LABELS = ['Tipe 1', 'Tipe 2', 'Tipe 3', 'Tipe 4', 'Tipe 5', 
          'Tipe 6', 'Tipe 7', 'Tipe 8', 'Tipe 9']

# --- 2. FUNGSI PREPROCESSING (JURUS HE KAMU) ---
def preprocess_image(image_stream):
    # Baca gambar dari memory (bukan path file)
    file_bytes = np.frombuffer(image_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Resize ke 128x128 (Sesuai training)
    img = cv2.resize(img, (128, 128))
    
    # --- METODE HE (Histogram Equalization) ---
    img = cv2.equalizeHist(img)
    img = cv2.bitwise_not(img) # Invert (Background Hitam)
    
    # Normalisasi & Reshape
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1) # Tambah channel
    img = np.expand_dims(img, axis=0)  # Tambah batch dimension
    
    return img

# --- 3. BIKIN PINTU MASUK (ROUTE) ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400
    
    file = request.files['file']
    
    try:
        # Proses Gambar
        processed_img = preprocess_image(file.stream)
        
        # Prediksi
        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result_label = LABELS[class_idx]
        
        # Kembalikan Hasil ke Temanmu (Format JSON)
        return jsonify({
            'status': 'success',
            'prediction': result_label,
            'confidence': f"{confidence*100:.2f}%",
            'message': 'Analisis Grafologi Selesai'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Jalankan Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)