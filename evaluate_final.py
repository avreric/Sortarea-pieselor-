import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# 1. CONFIGURARE CĂI (LOCALIZARE AUTOMATĂ)
# =========================================================
# Directorul curent unde se află acest script (proiect_rn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Numele modelului tău existent
MODEL_NAME = "model_inspectie_roti_dintate.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Căile către datele de test (numpy arrays)
TEST_IMAGES_PATH = os.path.join(BASE_DIR, "data", "test", "test_images.npy")
TEST_LABELS_PATH = os.path.join(BASE_DIR, "data", "test", "test_labels.npy")

# Folderele pentru rezultatele Etapei 6 (se creează automat)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def genereaza_livrabile_etapa6():
    print("--- Pornire Evaluare Finală Etapa 6 ---")

    # Verificăm dacă fișierele necesare există
    if not os.path.exists(MODEL_PATH):
        print(f"[EROARE] Nu am găsit modelul la: {MODEL_PATH}")
        return
    if not os.path.exists(TEST_IMAGES_PATH) or not os.path.exists(TEST_LABELS_PATH):
        print(f"[EROARE] Nu am găsit datele de test în data/test/")
        return

    # Cream folderele de ieșire dacă nu există
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DOCS_DIR, "results"), exist_ok=True)

    # -----------------------------------------------------
    # Pas 1: Încărcare Date și Model
    # -----------------------------------------------------
    print(f"[1/4] Încarc modelul: {MODEL_NAME}...")
    model = load_model(MODEL_PATH)
    
    X_test = np.load(TEST_IMAGES_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    print(f"      Am încărcat {len(X_test)} imagini de test.")

    # -----------------------------------------------------
    # Pas 2: Predicții și Confusion Matrix
    # -----------------------------------------------------
    print("[2/4] Generez predicții și Confusion Matrix...")
    prag = 0.60  # Pragul optim stabilit în app.py
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs >= prag).astype(int)

    # Generare grafic Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Conform', 'Defect'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - Model Optimizat (Prag {prag})')
    
    output_cm = os.path.join(DOCS_DIR, "confusion_matrix_optimized.png")
    plt.savefig(output_cm)
    plt.close()
    print(f"✅ Imagine salvată: {output_cm}")

    # -----------------------------------------------------
    # Pas 3: Calcul Metrici și Salvare JSON
    # -----------------------------------------------------
    print("[3/4] Calculez metricile finale...")
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / len(y_test)
    
    # False Negative Rate (Piese defecte ratate - foarte important!)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    # F1-Score pentru clasa defect
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_defect = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    final_metrics = {
        "model_evaluat": MODEL_NAME,
        "prag_decizie": prag,
        "test_accuracy": round(float(accuracy), 4),
        "f1_score_defect": round(float(f1_defect), 4),
        "false_negative_rate": round(float(fnr), 4),
        "total_test_samples": int(len(y_test)),
        "confusion_matrix": {
            "piese_bune_corecte": int(tn),
            "defecte_ratate": int(fn),
            "defecte_corecte": int(tp),
            "alarme_false": int(fp)
        }
    }

    output_json = os.path.join(RESULTS_DIR, "final_metrics.json")
    with open(output_json, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"✅ Metrici salvate: {output_json}")

    # -----------------------------------------------------
    # Pas 4: Grafic Evoluție (Baseline vs Optimizat)
    # -----------------------------------------------------
    print("[4/4] Generez graficul de evoluție...")
    etapa5_acc = 0.72  # Înlocuiește cu valoarea ta din Etapa 5 dacă e diferită
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Etapa 5 (Baseline)', 'Etapa 6 (Optimizat)'], [etapa5_acc, accuracy], color=['lightgrey', 'dodgerblue'])
    plt.ylabel('Acuratețe (%)')
    plt.title('Îmbunătățirea Performanței Sistemului')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_evo = os.path.join(DOCS_DIR, "results", "metrics_evolution.png")
    plt.savefig(output_evo)
    plt.close()
    print(f"✅ Grafic evoluție salvat: {output_evo}")

    print("\n--- ANALIZĂ COMPLETĂ ---")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Piese defecte ratate (FN): {fn}")
    print("---------------------------------------")

if __name__ == "__main__":
    genereaza_livrabile_etapa6()