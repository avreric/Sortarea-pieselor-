import os
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

MODEL_FILENAME = "model_inspectie_roti_dintate.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

IMAGE_SIZE = (128, 128)

DEFAULT_THRESHOLD = 0.60
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 0.99

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Nu găsesc modelul: {MODEL_PATH}\n"
        f"Pune '{MODEL_FILENAME}' în folderul proiectului (lângă app.py)."
    )

print("✅ Încarc modelul din:", MODEL_PATH)
model = load_model(MODEL_PATH)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clamp_threshold(v: float) -> float:
    if v < MIN_THRESHOLD:
        return MIN_THRESHOLD
    if v > MAX_THRESHOLD:
        return MAX_THRESHOLD
    return float(v)

def preprocess_image_from_bytes(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def predict_defect_probability(x):
    y = np.array(model.predict(x, verbose=0))

    if y.ndim == 2 and y.shape[1] == 1:   # sigmoid
        return float(y[0, 0])

    if y.ndim == 2 and y.shape[1] == 2:   # softmax -> defect = index 1
        return float(y[0, 1])

    return float(np.ravel(y)[0])

def save_result(original_bytes: bytes, original_filename: str, prob_defect: float, threshold: float, verdict: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(c for c in (original_filename or "upload") if c.isalnum() or c in ("-", "_", ".", " "))
    base = f"{ts}_{safe_name}".replace(" ", "_")

    img_path = os.path.join(RESULTS_DIR, base)
    with open(img_path, "wb") as f:
        f.write(original_bytes)

    log_path = os.path.join(RESULTS_DIR, f"{base}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"filename={original_filename}\n")
        f.write(f"prob_defect={prob_defect:.6f}\n")
        f.write(f"threshold={threshold:.6f}\n")
        f.write(f"verdict={verdict}\n")

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob_defect = None
    error = None
    threshold = DEFAULT_THRESHOLD

    if request.method == "POST":
        th_raw = (request.form.get("threshold", str(DEFAULT_THRESHOLD)) or "").strip()
        try:
            threshold = clamp_threshold(float(th_raw))
        except ValueError:
            threshold = DEFAULT_THRESHOLD

        if "image" not in request.files:
            error = "Nu s-a trimis niciun fișier."
            return render_template("index.html", result=result, prob_defect=prob_defect, error=error, threshold=round(threshold, 2))

        file = request.files["image"]
        if not file or file.filename == "":
            error = "Nu ai selectat nicio imagine."
            return render_template("index.html", result=result, prob_defect=prob_defect, error=error, threshold=round(threshold, 2))

        original_bytes = file.read()
        x = preprocess_image_from_bytes(original_bytes)
        if x is None:
            error = "Imagine invalidă sau nu a putut fi citită."
            return render_template("index.html", result=result, prob_defect=prob_defect, error=error, threshold=round(threshold, 2))

        prob_defect = predict_defect_probability(x)
        result = "DEFECTĂ" if prob_defect >= threshold else "CONFORMĂ"

        try:
            save_result(original_bytes, file.filename, prob_defect, threshold, result)
        except Exception as e:
            print("⚠️ Nu am putut salva în results/:", e)

    return render_template(
        "index.html",
        result=result,
        prob_defect=prob_defect,
        error=error,
        threshold=round(threshold, 2),
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
