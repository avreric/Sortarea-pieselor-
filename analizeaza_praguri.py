import os
import numpy as np
from tensorflow.keras.models import load_model

# ================== CĂI ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = os.path.join(BASE_DIR, "model_inspectie_roti_dintate.h5")


# ================== ÎNCĂRCARE DATE + MODEL ==================

def incarca_date_si_model():
    try:
        X_test = np.load(os.path.join(TEST_DIR, "test_images.npy"))
        y_test = np.load(os.path.join(TEST_DIR, "test_labels.npy"))
    except FileNotFoundError as e:
        print("[EROARE] Nu am găsit fișierele .npy pentru test:", e)
        return None, None, None

    if not os.path.exists(MODEL_PATH):
        print("[EROARE] Nu am găsit modelul .h5 la:", MODEL_PATH)
        return None, None, None

    model = load_model(MODEL_PATH)

    print(f"[INFO] Am încărcat {len(X_test)} imagini de test.")
    valori, cnt = np.unique(y_test, return_counts=True)
    print("[INFO] Distribuția etichetelor în TEST:")
    for v, c in zip(valori, cnt):
        print(f"   clasa {int(v)}: {c} imagini")
    print()

    return X_test, y_test, model


# ================== METRICE ==================

def calculeaza_metrici(y_true, y_pred):
    """
    y_true: valori reale (0 / 1)
    y_pred: valori prezise (0 / 1)
    Returnează (acc, TP, TN, FP, FN, recall_defecte)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total = len(y_true)
    acc = (TP + TN) / total if total > 0 else 0.0

    # „Recall” pentru defecte = câte defecte reale a prins
    if (TP + FN) > 0:
        recall_defecte = TP / (TP + FN)
    else:
        recall_defecte = 0.0

    return acc, TP, TN, FP, FN, recall_defecte


# ================== ANALIZĂ ==================

def analizeaza_praguri():
    X_test, y_test, model = incarca_date_si_model()
    if X_test is None:
        return

    # model.predict -> probabilitate că piesa e DEFECTĂ
    prob_defect = model.predict(X_test).flatten()

    print("Probabilități model (piesa e DEFECTĂ) pentru imaginile de test:")
    for i, (p, real) in enumerate(zip(prob_defect, y_test)):
        print(f"  Imaginea {i}: p_defect={p:.4f}, etichetă_reală={int(real)}")
    print()

    praguri = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70]

    print("==== REZULTATE PE PRAGURI DIFERITE ====\n")
    for thr in praguri:
        pred_labels = (prob_defect >= thr).astype(int)
        acc, TP, TN, FP, FN, recall_defecte = calculeaza_metrici(y_test, pred_labels)

        print(f"PRAG = {thr:.2f}")
        print(f"  Acuratețe totală: {acc * 100:.2f}%")
        print(f"  TP (defecte prinse corect):     {TP}")
        print(f"  FN (defecte scăpate ca bune):   {FN}")
        print(f"  FP (bune marcate defecte):      {FP}")
        print(f"  TN (bune recunoscute corect):   {TN}")
        print(f"  Recall defecte (TP / (TP+FN)):  {recall_defecte * 100:.2f}%")
        print("-" * 45)


if __name__ == "__main__":
    analizeaza_praguri()
