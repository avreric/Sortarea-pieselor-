import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- PARAMETRII PROIECTULUI ȘI CĂI ---
# Determinăm calea absolută a rădăcinii proiectului, asumând că rulezi din Command Prompt
# unde CWD este proiect_rn, sau folosim o metodă similară celei de la pre-procesare.

# Definirea căii relative de la directorul de rulare la directorul 'data'
DATA_DIR_NAME = 'data' 

# Calea absolută a directorului unde se află datele .npy (e.g., .../proiect_rn/data)
# Această logică este cea mai sigură, bazată pe locația scriptului de train.
DIRECTOR_CURENT_SCRIPT = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, DATA_DIR_NAME) 

# Căi de încărcare:
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Calea de salvare a modelului: modelul se va salva lângă script, nu în data/
MODEL_PATH = os.path.join(DIRECTOR_CURENT_SCRIPT, 'model_inspectie_roti_dintate.h5')

# Setări model
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1) # Alb-Negru
NUMAR_CLASE = 1 
EPOCI = 15 

def incarca_date_npy():
    """Încarcă toate array-urile .npy pentru antrenare, validare și test."""
    try:
        X_train = np.load(os.path.join(TRAIN_DIR, 'train_images.npy'))
        y_train = np.load(os.path.join(TRAIN_DIR, 'train_labels.npy'))
        X_val = np.load(os.path.join(VAL_DIR, 'validation_images.npy'))
        y_val = np.load(os.path.join(VAL_DIR, 'validation_labels.npy'))
        X_test = np.load(os.path.join(TEST_DIR, 'test_images.npy'))
        y_test = np.load(os.path.join(TEST_DIR, 'test_labels.npy'))

        print(f"Date încărcate. Train: {len(X_train)} imagini, Val: {len(X_val)} imagini.")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError as e:
        print(f"Eroare la încărcarea fișierelor .npy: {e}")
        print("Asigură-te că directorul 'data' există și conține subfolderele 'train', 'validation', 'test' cu fișierele .npy!")
        return None, None, None, None, None, None

def defineste_si_antreneaza_modelul(X_train, y_train, X_val, y_val):
    """Definește arhitectura CNN și antrenează modelul."""
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), 
        
        Dense(NUMAR_CLASE, activation='sigmoid') 
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    
    print("\n--- Începe Antrenarea ---")
    model.fit(
        X_train, y_train,
        epochs=EPOCI,
        validation_data=(X_val, y_val)
    )

    return model

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = incarca_date_npy()
    
    if X_train is not None:
        model = defineste_si_antreneaza_modelul(X_train, y_train, X_val, y_val)
        
        # Evaluare pe setul de test
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n--- Evaluare Finală ---")
        print(f"Acuratețe pe setul de test: {acc*100:.2f}%")
        
        # Salvarea modelului
        model.save(MODEL_PATH)
        print(f"Modelul a fost salvat ca: {MODEL_PATH}")