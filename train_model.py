import os
import numpy as np
from tensorflow.keras.models import load_model

# --- Configurare rapidă ---
# Am lăsat path-urile aici să fie ușor de schimbat dacă mutăm folderul
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_FILE = os.path.join(BASE_DIR, "model_inspectie_roti_dintate.h5")

def get_data():
    """Încarcă datele de test. Aruncă eroare dacă fișierele .npy lipsesc."""
    x_path = os.path.join(TEST_DATA_DIR, "test_images.npy")
    y_path = os.path.join(TEST_DATA_DIR, "test_labels.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[!] Eroare: Nu am găsit fișierele npy în {TEST_DATA_DIR}")
        return None, None

    X_test = np.load(x_path)
    y_test = np.load(y_path)
    
    # Un mic print de control
    print(f"-> Am încărcat {len(X_test)} imagini pentru testare.")
    return X_test, y_test

def run_evaluation():
    # 1. Încărcare model
    if not os.path.exists(MODEL_FILE):
        print(f"Eroare: Modelul {MODEL_FILE} nu e de găsit!")
        return

    print("Se încarcă modelul Keras... (poate dura puțin)")
    model = load_model(MODEL_FILE)

    # 2. Încărcare date
    X_test, y_test = get_data()
    if X_test is None:
        return

    # 3. Quick stats
    classes, counts = np.unique(y_test, return_counts=True)
    print("\n--- Distribuție Clase ---")
    for cls, count in zip(classes, counts):
        tag = "OK" if cls == 0 else "Defect"
        print(f" Clasa {int(cls)} ({tag}): {count} mostre")
    print("------------------------\n")

    # Aici poți adăuga model.evaluate(X_test, y_test) dacă vrei rezultatele direct
    return model, X_test, y_test

if __name__ == "__main__":
    model, x, y = run_evaluation()
    if model:
        print("Gata. Totul e pregătit pentru predicții.")