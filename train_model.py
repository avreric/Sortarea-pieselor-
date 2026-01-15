import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ================== SETĂRI GENERALE ==================

# Numele directorului de date relativ la acest script
DATA_DIR_NAME = 'data'

# Directorul unde se află acest script (proiect_rn)
DIRECTOR_CURENT_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Calea absolută către data/
DATA_DIR = os.path.join(DIRECTOR_CURENT_SCRIPT, DATA_DIR_NAME)

# Subfoldere cu .npy
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Calea unde se va salva modelul antrenat
MODEL_PATH = os.path.join(DIRECTOR_CURENT_SCRIPT, 'model_inspectie_roti_dintate.h5')

# Imaginile tale sunt 128x128 alb-negru
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Număr de epoci maxime (EarlyStopping va opri mai devreme dacă e cazul)
EPOCI_MAXIME = 50
BATCH_SIZE = 4  # batch mic pentru dataset mic


# ================== ÎNCĂRCARE DATE ==================

def incarca_date_npy():
    """Încarcă toate array-urile .npy pentru antrenare, validare și test."""
    try:
        X_train = np.load(os.path.join(TRAIN_DIR, 'train_images.npy'))
        y_train = np.load(os.path.join(TRAIN_DIR, 'train_labels.npy'))

        X_val = np.load(os.path.join(VAL_DIR, 'validation_images.npy'))
        y_val = np.load(os.path.join(VAL_DIR, 'validation_labels.npy'))

        X_test = np.load(os.path.join(TEST_DIR, 'test_images.npy'))
        y_test = np.load(os.path.join(TEST_DIR, 'test_labels.npy'))

        print(f"[INFO] Date încărcate cu succes.")
        print(f"       Train: {len(X_train)} imagini")
        print(f"       Val:   {len(X_val)} imagini")
        print(f"       Test:  {len(X_test)} imagini\n")

        # Verificăm distribuția etichetelor (0 / 1)
        valori, cnt = np.unique(y_train, return_counts=True)
        print(f"[INFO] Distribuție etichete în TRAIN:")
        for v, c in zip(valori, cnt):
            print(f"       Clasa {int(v)}: {c} imagini")
        print()

        return X_train, y_train, X_val, y_val, X_test, y_test

    except FileNotFoundError as e:
        print(f"[EROARE] Nu am găsit unul din fișierele .npy: {e}")
        print("         Verifică să existe folderele 'train', 'validation', 'test' în data/")
        return None, None, None, None, None, None


# ================== DEFINIRE MODEL ==================

def defineste_modelul():
    """Definește un model CNN mai mic, potrivit pentru set mic de date."""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),

        Dense(1, activation='sigmoid')  # ieșire binară: 0 = conformă, 1 = defectă
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n========== ARHITECTURA MODELULUI ==========")
    model.summary()
    print("===========================================\n")

    return model


# ================== ANTRENARE CU AUGMENTARE ==================

def antreneaza_modelul(model, X_train, y_train, X_val, y_val):
    """Antrenează modelul folosind augmentare de date și EarlyStopping."""

    # Augmentare de date – modificări mici, realiste
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3)   # robust la umbre / luminozitate diferită
    )

    datagen.fit(X_train)

    # EarlyStopping ca să nu supra-antrenăm
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    print("[INFO] Începe antrenarea modelului...\n")

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCI_MAXIME,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    print("\n[INFO] Antrenarea s-a încheiat.")
    return history


# ================== EVALUARE ȘI SALVARE ==================

def evalueaza_modelul(model, X_test, y_test):
    """Evaluează modelul pe setul de test."""
    print("\n[INFO] Evaluez modelul pe setul de TEST...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[REZULTAT] Loss (test): {loss:.4f}")
    print(f"[REZULTAT] Acuratețe pe setul de test: {acc * 100:.2f}%\n")


def salveaza_modelul(model):
    """Salvează modelul în format .h5 lângă acest script."""
    model.save(MODEL_PATH)
    print(f"[INFO] Modelul a fost salvat la: {MODEL_PATH}")


# ================== MAIN ==================

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = incarca_date_npy()

    if X_train is None:
        exit(1)

    model = defineste_modelul()
    history = antreneaza_modelul(model, X_train, y_train, X_val, y_val)
    evalueaza_modelul(model, X_test, y_test)
    salveaza_modelul(model)
