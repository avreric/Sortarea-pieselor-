import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Luăm IMAGE_SIZE din scriptul de preprocesare, ca să fim siguri că e identic
from data.importcv2 import IMAGE_SIZE

# ================== SETĂRI GENERALE ==================

# Directorul în care se află acest fișier (proiect_rn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Căi posibile către model:
# 1) modelul antrenat în Etapa 5 (recomandat)
# 2) vechiul model, ca fallback
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.h5")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "model_inspectie_roti_dintate.h5")

# 0 = piesă conformă, 1 = piesă defectă
ETICHETE = {
    0: "CONFORMĂ (BUNĂ)",
    1: "DEFECTĂ"
}

# Prag de decizie:
# dacă probabilitatea că piesa e DEFECTĂ >= THRESHOLD -> o declarăm DEFECTĂ
THRESHOLD = 0.5  # poți pune 0.6 în aplicație dacă vrei să prinzi mai multe defecte

# ================== FUNCȚII HELPER ==================

def preproceseaza_imagine_test(cale_imagine: str):
    """
    Încarcă imaginea, o convertește în alb-negru, o redimensionează la IMAGE_SIZE
    și o normalizează în [0, 1].
    Returnează un array (1, H, W, 1) gata de băgat în model.
    """
    if not os.path.exists(cale_imagine):
        print(f"[EROARE] Imaginea nu a fost găsită la calea: {cale_imagine}")
        return None

    img = cv2.imread(cale_imagine, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[EROARE] Nu s-a putut citi imaginea. Verifică formatul / permisiunile.")
        return None

    # Redimensionare la aceeași mărime ca în importcv2.py
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    img_array = img_to_array(img_resized)          # (H, W, 1)
    img_array = img_array.astype("float32") / 255.0
    img_final = np.expand_dims(img_array, axis=0)  # (1, H, W, 1)

    return img_final


def incarca_model():
    """Încarcă modelul salvat de pe disc (trained_model.h5 sau fallback)."""
    if os.path.exists(TRAINED_MODEL_PATH):
        path = TRAINED_MODEL_PATH
        print(f"[INFO] Încarc modelul antrenat: {path}")
    elif os.path.exists(FALLBACK_MODEL_PATH):
        path = FALLBACK_MODEL_PATH
        print(f"[WARN] models/trained_model.h5 nu există, folosesc: {path}")
    else:
        print("[EROARE] Nu am găsit nici models/trained_model.h5, nici model_inspectie_roti_dintate.h5.")
        print("        Asigură-te că ai rulat train_model.py sau că fișierul .h5 există.")
        return None

    try:
        model = load_model(path)
        print("[INFO] Modelul AI a fost încărcat cu succes.")
        return model
    except Exception as e:
        print(f"[EROARE] A apărut o problemă la încărcarea modelului:\n        {e}")
        return None


# ================== FLUX PRINCIPAL ==================

def clasifica_piesa():
    """Funcția principală: cere o cale de imagine și afișează verdictul modelului."""
    model = incarca_model()
    if model is None:
        return

    cale_imagine_test = input(
        "\nIntrodu calea completă a imaginii de test "
        "(ex: C:\\Imagini\\roata_uzata.jpg): "
    ).strip()

    if not cale_imagine_test:
        print("[EROARE] Nu ai introdus nicio cale de fișier.")
        return

    imagine_procesata = preproceseaza_imagine_test(cale_imagine_test)
    if imagine_procesata is None:
        return

    # Probabilitatea (0..1) că piesa e DEFECTĂ (clasa 1)
    probabilitate_defecta = float(model.predict(imagine_procesata)[0][0])

    print(f"\nProbabilitatea brută (model) că piesa este DEFECTĂ: "
          f"{probabilitate_defecta:.4f}")

    if probabilitate_defecta >= THRESHOLD:
        clasa_pred = 1
        incredere = probabilitate_defecta          # încredere că E defectă
    else:
        clasa_pred = 0
        incredere = 1.0 - probabilitate_defecta    # încredere că NU e defectă

    rezultat_clasa = ETICHETE[clasa_pred]

    print("\n" + "=" * 40)
    print("REZULTAT CLASIFICARE:")
    print(f"Piesa este clasificată ca: **{rezultat_clasa}**")
    print(f"Încredere model (pentru acest verdict): {incredere * 100:.2f}%")
    print(f"(Prag de decizie folosit: {THRESHOLD:.2f})")
    print("=" * 40)


if __name__ == '__main__':
    clasifica_piesa()
