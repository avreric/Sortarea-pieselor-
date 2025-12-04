import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. PARAMETRI ȘI CĂI (BAZATE PE LOCAȚIA SCRIPTULUI) ---

# Calculăm calea absolută a directorului unde se află acest fișier (e.g., .../project_rn/data/)
# Aceasta asigură că scriptul funcționează indiferent de CWD-ul terminalului.
DIRECTOR_CURENT_SCRIPT = os.path.dirname(os.path.abspath(__file__)) 

# Setări generale
IMAGE_SIZE = (128, 128) 

# CĂI CORECTATE FINAL:
RAW_DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'raw') 
PROCESSED_DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'processed')
TRAIN_DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'train')
VALIDATION_DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'validation')
TEST_DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'test')

# NOU: Director pentru salvarea copiilor vizuale (JPEG/JPG)
IMAGINI_VIZUALE_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, 'processed_images_vizual')

CLASSES = ['piese conforme', 'piese defecte']
TEST_SIZE = 0.15      
VALIDATION_SIZE = 0.15  

# --- 2. ÎNCĂRCAREA ȘI PRE-PROCESAREA IMAGINILOR ---

def load_and_preprocess_images():
    data = []
    labels = []
    
    # Creează directorul pentru imaginile vizuale (JPEG)
    try:
        os.makedirs(IMAGINI_VIZUALE_DIR, exist_ok=True)
        print(f"Directorul vizual creat/verificat: {IMAGINI_VIZUALE_DIR}")
    except Exception as e:
        print(f"Eroare la crearea directorului vizual: {e}")
        return data, labels
    
    for class_index, class_name in enumerate(CLASSES):
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_path):
            print(f"Eroare: Calea '{class_path}' nu există. Ignor.")
            continue
            
        print(f"Procesez folderul: {class_name} (Clasa {class_index})")
        
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                
                # 1. Încarcă imaginea (Alb-Negru)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # 2. Micșorarea Rezoluției 
                img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                
                # --- PAS SALVARE VIZUALĂ (JPEG) ---
                # Salvăm imaginea redimensionată (0-255) ca fișier JPEG
                vizual_filename = f'{class_name}_{filename}' 
                vizual_path = os.path.join(IMAGINI_VIZUALE_DIR, vizual_filename)
                
                cv2.imwrite(vizual_path, img_resized)
                # --- SFÂRȘIT SALVARE VIZUALĂ ---
                
                # 3. Normalizarea (pentru AI)
                img_normalized = img_resized / 255.0
                
                # 4. Adaugă dimensiunea canalului (1)
                img_final = np.expand_dims(img_normalized, axis=-1) 
                
                data.append(img_final)
                labels.append(class_index)
                
    X = np.array(data, dtype='float32')
    y = np.array(labels, dtype='int32')
    
    print(f"\nProcesare finalizată. Total imagini procesate: {len(X)}")
    print(f"Imaginile JPEG vizuale au fost salvate în: {IMAGINI_VIZUALE_DIR}")
    return X, y

def save_numpy_data(X, y, directory, name):
    """ Salvează setul de date X și y în directorul specificat. """
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, f'{name}_images.npy'), X)
    np.save(os.path.join(directory, f'{name}_labels.npy'), y)
    print(f"Salvat setul {name} ({len(X)} imagini) în {directory}")

def save_all_processed_data(X, y):
    """ Salvează toate datele procesate într-un singur loc (pentru arhivă). """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'images.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'labels.npy'), y)
    print(f"Datele complete procesate au fost salvate în {PROCESSED_DATA_DIR}")

# --- 3. ÎMPĂRȚIREA DATELOR (DATA SPLITTING) ---

def split_and_save_data(X, y):
    """ Împarte setul X și y în Train, Validation și Test. """
    if len(X) == 0:
        print("Nu există date de împărțit.")
        return
        
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    validation_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_ratio, random_state=42, stratify=y_temp
    )
    
    print("\n--- Rezultate Împărțire ---")
    
    save_numpy_data(X_train, y_train, TRAIN_DATA_DIR, 'train')
    save_numpy_data(X_val, y_val, VALIDATION_DATA_DIR, 'validation')
    save_numpy_data(X_test, y_test, TEST_DATA_DIR, 'test')
    
    print("-" * 30)
    print(f"Total imagini: {len(X)}")
    print(f"Set Antrenare: {len(X_train)} ({len(X_train)/len(X):.2%})")
    print(f"Set Validare: {len(X_val)} ({len(X_val)/len(X):.2%})")
    print(f"Set Test: {len(X_test)} ({len(X_test)/len(X):.2%})")
    print("-" * 30)

# --- 4. RULAREA PRINCIPALĂ A SCRIPTULUI ---

if __name__ == '__main__':
    print("--- Pornesc Pre-procesarea Datelor ---")
    
    X_processed, y_labels = load_and_preprocess_images()
    
    if len(X_processed) > 0:
        save_all_processed_data(X_processed, y_labels)
        split_and_save_data(X_processed, y_labels)
    else:
        print("\n[STOP] Nu s-au găsit imagini de procesat. Verifică structura 'raw/piese conforme' și 'raw/piese defecte'.")
        