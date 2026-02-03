import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow import keras

from importcv2 import TEST_DATA_DIR   # e în același folder cu acest fișier

# ROOT_DIR = proiect_rn
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DATA_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_data():
    X_test = np.load(os.path.join(TEST_DATA_DIR, "test_images.npy"))
    y_test = np.load(os.path.join(TEST_DATA_DIR, "test_labels.npy"))
    print("Test:", X_test.shape, y_test.shape)
    return X_test, y_test

def load_model():
    """Încarcă models/trained_model.h5; dacă nu există, folosește vechiul model_inspectie_roti_dintate.h5"""
    trained_path = os.path.join(MODELS_DIR, "trained_model.h5")
    fallback_path = os.path.join(ROOT_DIR, "model_inspectie_roti_dintate.h5")

    if os.path.exists(trained_path):
        print(f"[INFO] Încarc {trained_path}")
        return keras.models.load_model(trained_path)

    if os.path.exists(fallback_path):
        print(f"[WARN] Nu există models/trained_model.h5, folosesc {fallback_path}")
        return keras.models.load_model(fallback_path)

    raise FileNotFoundError(
        "Nu există nici models/trained_model.h5, nici model_inspectie_roti_dintate.h5 în rădăcina proiectului."
    )

def main():
    X_test, y_test = load_test_data()
    model = load_model()

    # Probabilitatea că piesa este DEFECTĂ
    y_proba = model.predict(X_test).ravel()

    # Prag standard pentru metrici. În aplicație poți folosi alt prag (ex. 0.6).
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")

    print("==================================")
    print(f" Test Accuracy:  {acc:.4f}")
    print(f" Test F1 macro:  {f1:.4f}")
    print(f" Test Precision: {prec:.4f}")
    print(f" Test Recall:    {rec:.4f}")
    print("==================================")

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "threshold_used": float(threshold),
    }

    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("[OK] Metrici salvate în results/test_metrics.json")

if __name__ == "__main__":
    main()
