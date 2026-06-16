"""
app.py
Flask Web Application - Klasifikasi Motif Batik Indonesia
Jalankan: python app.py
"""

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import uuid
from PIL import Image

# ============================================================
# KONFIGURASI
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # Maks 16 MB
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.secret_key                   = 'batik-cnn-secret-2024'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH         = 'model/batik_model.h5'
LABELS_PATH        = 'model/class_names.txt'
IMG_SIZE           = (128, 128)

# ============================================================
# DESKRIPSI MOTIF BATIK
# ============================================================
BATIK_INFO = {
    "batik_parang": {
        "nama"   : "Batik Parang",
        "asal"   : "Yogyakarta / Solo, Jawa Tengah",
        "makna"  : "Melambangkan kekuatan, ketangkasan, dan semangat yang tidak pernah putus. "
                   "Motif parang berasal dari kata 'pereng' (lereng) yang menggambarkan ombak laut.",
        "warna"  : "Coklat sogan, hitam, krem",
        "emoji"  : "⚔️"
    },
    "batik_kawung": {
        "nama"   : "Batik Kawung",
        "asal"   : "Yogyakarta, Jawa Tengah",
        "makna"  : "Melambangkan kesucian, keadilan, dan harapan agar manusia selalu ingat "
                   "akan asal usulnya. Motif kawung menyerupai buah aren (kawung).",
        "warna"  : "Putih, hitam, coklat",
        "emoji"  : "🌀"
    },
    "batik_mega_mendung": {
        "nama"   : "Batik Mega Mendung",
        "asal"   : "Cirebon, Jawa Barat",
        "makna"  : "Melambangkan kesabaran dan ketenangan hati. Terinspirasi dari awan (mega) "
                   "yang membawa kesejukan dan hujan sebagai berkah.",
        "warna"  : "Biru tua ke biru muda, merah",
        "emoji"  : "☁️"
    },
    "batik_truntum": {
        "nama"   : "Batik Truntum",
        "asal"   : "Solo, Jawa Tengah",
        "makna"  : "Melambangkan cinta yang bersemi kembali dan kesetiaan. Biasa dipakai "
                   "orang tua pengantin sebagai simbol bimbingan dan kasih sayang.",
        "warna"  : "Hitam dengan bintang-bintang putih/kuning",
        "emoji"  : "⭐"
    },
    "batik_sekar_jagad": {
        "nama"   : "Batik Sekar Jagad",
        "asal"   : "Solo & Yogyakarta, Jawa Tengah",
        "makna"  : "Melambangkan keberagaman dan keindahan dunia. 'Sekar' berarti bunga, "
                   "'Jagad' berarti dunia — menggambarkan keindahan alam semesta.",
        "warna"  : "Beragam warna cerah",
        "emoji"  : "🌸"
    }
}

# ============================================================
# LOAD MODEL
# ============================================================
model       = None
class_names = []

def load_model_and_labels():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        print("[INFO] Memuat model CNN...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"[INFO] Model siap! Kelas: {class_names}")
        return True
    else:
        print("[WARNING] Model belum tersedia. Jalankan train_model.py terlebih dahulu.")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 128, 128, 3)
    return img_array

def get_prediction(img_path):
    img_array  = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_idx   = int(np.argmax(predictions[0]))
    confidence      = float(np.max(predictions[0])) * 100
    predicted_label = class_names[predicted_idx]

    # Top-3 prediksi
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {
            "label"      : class_names[i],
            "nama"       : BATIK_INFO.get(class_names[i], {}).get("nama", class_names[i]),
            "confidence" : round(float(predictions[0][i]) * 100, 2)
        }
        for i in top3_idx
    ]

    info = BATIK_INFO.get(predicted_label, {
        "nama" : predicted_label.replace("_", " ").title(),
        "asal" : "-",
        "makna": "-",
        "warna": "-",
        "emoji": "🎨"
    })

    return {
        "label"     : predicted_label,
        "confidence": round(confidence, 2),
        "info"      : info,
        "top3"      : top3
    }

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    model_ready = model is not None
    return render_template('index.html', model_ready=model_ready)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html',
                               error="Model belum tersedia. Jalankan train_model.py terlebih dahulu.",
                               model_ready=False)

    if 'file' not in request.files:
        return render_template('result.html', error="Tidak ada file yang diunggah.", model_ready=True)

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="Tidak ada file yang dipilih.", model_ready=True)

    if not allowed_file(file.filename):
        return render_template('result.html',
                               error="Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.",
                               model_ready=True)

    # Simpan file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    ext       = file.filename.rsplit('.', 1)[1].lower()
    filename  = f"{uuid.uuid4().hex}.{ext}"
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Prediksi
    result          = get_prediction(filepath)
    result['image'] = url_for('static', filename=f'uploads/{filename}')

    return render_template('result.html', result=result, model_ready=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk integrasi eksternal"""
    if model is None:
        return jsonify({"error": "Model belum tersedia"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = get_prediction(filepath)
    return jsonify(result)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    load_model_and_labels()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )