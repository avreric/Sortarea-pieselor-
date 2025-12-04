import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- PARAMETRII ȘI CĂI ---
RUTINĂ_PROIECT = r'C:\Users\avrer\Desktop\proiect_rn' 
MODEL_PATH = os.path.join(RUTINĂ_PROIECT, 'model_inspectie_roti_dintate.h5')

IMAGE_SIZE = (128, 128)
ETICHETE = {0: "CONFORMĂ (BUNĂ)", 1: "DEFECTĂ"} # 0 și 1 depind de ordinea alfabetică a subfolderelor


def preproceseaza_imagine_test(cale_imagine):
    """Încarcă imaginea, o face alb-negru, o redimensionează și o normalizează."""
    
    # Verifică existența fișierului
    if not os.path.exists(cale_imagine):
        print(f"Eroare: Imaginea nu a fost găsită la calea: {cale_imagine}")
        return None
        
    # 1. Încarcă imaginea ca alb-negru (Grayscale) și redimensionează
    img = cv2.imread(cale_imagine, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Eroare: Nu s-a putut citi imaginea.")
        return None
        
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Convertește la array NumPy și normalizează
    img_array = img_to_array(img_resized)
    img_array /= 255.0

    # 3. Adaugă dimensiunea pentru batch și canal (128, 128, 1) -> (1, 128, 128, 1)
    img_final = np.expand_dims(img_array, axis=0) 
    
    return img_final

def clasifica_piesa():
    """Funcția principală pentru a introduce o cale și a face predicția."""
    try:
        # 1. Încarcă modelul salvat
        model = load_model(MODEL_PATH)
        print("Modelul AI a fost încărcat cu succes.")
        
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        print(f"Asigură-te că fișierul model_inspectie_roti_dintate.h5 există la calea: {MODEL_PATH}")
        return

    # 2. Cere calea imaginii de la utilizator
    cale_imagine_test = input("\nIntrodu calea completă a imaginii de test (ex: C:\\Imagini\\roata_uzata.jpg): ")

    # 3. Pre-procesează imaginea
    imagine_procesata = preproceseaza_imagine_test(cale_imagine_test)
    
    if imagine_procesata is None:
        return

    # 4. Realizează predicția
    # Rezultatul este o probabilitate între 0 și 1
    probabilitate = model.predict(imagine_procesata)[0][0]
    
    # 5. Interpretează rezultatul
    if probabilitate >= 0.5:
        rezultat_clasa = ETICHETE[1] # Defectă
        prob_finala = probabilitate
    else:
        rezultat_clasa = ETICHETE[0] # Conformă
        prob_finala = 1 - probabilitate
        
    print("\n" + "=" * 40)
    print(f"REZULTAT CLASIFICARE:")
    print(f"Piesa este clasificată ca: **{rezultat_clasa}**")
    print(f"Încredere model: {prob_finala*100:.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    clasifica_piesa()