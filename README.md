# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Avram Eric Mario]  
**Proiect:** Sortarea Pieselor  
**Data:** [20.11.2025]

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, având ca scop pregătirea setului de date pentru proiectul **„Detectarea Defectelor în Piese Mecanice"**. Obiectivul final este antrenarea unui model de tip **Vision Transformer (ViT)** capabil să clasifice automat imaginile industriale în două categorii: *Piese Bune* și *Piese Defecte*[cite: 11, 32].

Procesul respectă fluxul standard de Machine Learning: achiziție, analiză exploratorie (EDA), curățare și preprocesare (inclusiv augmentare pentru a compensa necesitatea unui set mare de date ).

---

## 1. Structura Repository-ului Github (versiunea Etapei 3)

Structura a fost adaptată pentru un proiect de Computer Vision implementat în Python/PyTorch:
```
  Defect-Detection-ViT/
├── README.md              # Documentația curentă
├── docs/
│   └── dataset/info.md    # Detalii despre sursa imaginilor și etichete
├── data/
│   ├── raw/               # Imaginile originale organizate (ex: folder 'defective', folder 'ok')
│   ├── processed/         # Imaginile redimensionate și normalizate (numpy arrays / tensors)
│   ├── train/             # Sub-setul de antrenare
│   ├── validation/        # Sub-setul de validare
│   └── test/              # Sub-setul de testare
├── src/
│   ├── preprocessing.py   # Scripturi pentru resize, normalizare, augmentare (OpenCV/Albumentations)
│   ├── analysis.py        # Scripturi pentru generarea histogramelor și EDA
│   └── utils.py           # Funcții auxiliare
├── config/
│   └── config.yaml        # Parametri (img_size: 224, batch_size: 32, etc.)
└── requirements.txt       # Dependențe: PyTorch, OpenCV, Matplotlib, NumPy [cite: 52]

##2. Descrierea Setului de Date

###2.1 Sursa datelor

    Origine: Dataset public reprezentativ pentru piese turnate (ex: Casting Product Image Data for Quality Inspection), simulant o linie de producție reală.

Modul de achiziție: Imagini capturate prin camere video industriale (vedere de sus), iluminare controlată.

    Perioada / condițiile colectării: Imagini statice, format grayscale sau RGB, focalizate pe piesa de interes.

###2.2 Caracteristicile dataset-ului

    Număr total de observații: [Ex: 7,348 imagini] (Completează cu numărul real din dataset-ul ales).

    Număr de clase: 2 (Clasificare binară: ok_front vs def_front).

Tipuri de date: Imagini (Matrici de pixeli).

Format fișiere: ☐ CSV / ☐ TXT / ☐ JSON / ☑ PNG/JPG / ☐ Altele.

###2.3 Descrierea caracteristicilor (Atributele Imaginilor)

Deoarece lucrăm cu date nestructurate (imagini), caracteristicile sunt definite de proprietățile vizuale și metadate:

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|--------------------|---------|-------------|---------------|--------------------|
| Image_Height | numeric | pixeli | Înălțimea imaginii originale | [ex: 300] |
| Image_Width | numeric | pixeli | Lățimea imaginii originale | [ex: 300] |
| Channels | numeric | - | Canale de culoare (1=Gray, 3=RGB) | {1, 3} |
| Pixel_Intensity | numeric | - | Valoarea intensității unui pixel | 0 – 255 |
| **Label** (Target) | categorial | - | Clasificarea piesei (defectă/bună) | {0 (Bun), 1 (Defect)} |


##3. Analiza Exploratorie a Datelor (EDA) – Sintetic

###3.1 Statistici descriptive aplicate

    Distribuția claselor: S-a calculat numărul de imagini pentru fiecare clasă pentru a verifica echilibrul setului de date.

    Analiza dimensiunilor: Verificarea consistenței rezoluției imaginilor (toate au aceeași dimensiune sau necesită resize?).

    Distribuția intensității pixelilor: Histograme ale valorilor medii ale pixelilor pentru a detecta imagini prea întunecate sau supraexpuse.

###3.2 Analiza calității datelor

    Detectarea imaginilor corupte: Verificarea fișierelor care nu pot fi deschise cu biblioteca OpenCV.

Verificarea duplicatelor: Identificarea imaginilor identice care ar putea duce la data leakage între Train și Test.

Analiza vizuală: Vizualizarea randomizată a mostrelor pentru a confirma etichetarea corectă (ex: fisuri vizibile pe piesele etichetate ca "Defect").

###3.3 Probleme identificate

    Dezechilibru de clasă (Class Imbalance): S-a observat că numărul pieselor "Bune" este mai mare decât al celor "Defecte" (situație tipică în industrie).

        Impact: Modelul ar putea tinde să prezică mereu "Piesă Bună".

    Variații de poziție: Piesele nu sunt centrate perfect în toate imaginile.

    Dimensiune limitată a setului de date: Numărul de imagini cu defecte specifice este mic, ceea ce necesită tehnici de augmentare.


##4. Preprocesarea Datelor

###4.1 Curățarea datelor

    Eliminare fișiere corupte: S-au șters imaginile care aveau dimensiunea 0kb sau format invalid.

    Filtrare: S-au păstrat doar imaginile care conțin piesa completă în cadru.

###4.2 Transformarea caracteristicilor

Pentru a pregăti imaginile pentru Vision Transformer (ViT), s-au aplicat următoarele transformări folosind torchvision.transforms:

    Redimensionare (Resize): Toate imaginile au fost aduse la dimensiunea standard de 224x224 pixeli (cerință standard ViT).

    Normalizare: Valorile pixelilor (0-255) au fost scalate în intervalul [0, 1] și apoi normalizate folosind media și deviația standard (ex: ImageNet stats: mean=[0.485, ...], std=[0.229, ...]).

    Augmentarea Datelor (Data Augmentation): Pentru a combate limitările setului de date, s-au aplicat pe setul de antrenare:

        Random Horizontal Flip

        Random Rotation (±10 grade)

        Ajustări ușoare de luminozitate.

###4.3 Structurarea seturilor de date

Setul de date a fost împărțit aleatoriu, dar stratificat (păstrând proporția defect/bun), în:

    70% – Train: Pentru antrenarea parametrilor modelului ViT.

    15% – Validation: Pentru monitorizarea performanței și ajustarea hiperparametrilor.

    15% – Test: Pentru evaluarea finală obiectivă.

###4.4 Salvarea rezultatelor preprocesării

    Imaginile brute au rămas în data/raw/ pentru siguranță.

    Scripturile de Dataloaders din PyTorch au fost configurate pentru a citi și transforma datele în timp real (on-the-fly) pentru a economisi spațiu pe disc.

##5. Fișiere Generate în Această Etapă

    data/raw/ – Folderul cu dataset-ul original descărcat.

    src/preprocessing/data_loader.py – Codul Python pentru încărcarea și augmentarea imaginilor.

    src/analysis/eda_notebook.ipynb – Notebook Jupyter cu graficele distribuțiilor și exemple de imagini.

    data/split/ – Fișiere text sau CSV care conțin listele de fișiere pentru train/val/test (pentru reproductibilitate).

##6. Stare Etapă

    [x] Structură repository configurată conform cerințelor.

    [x] Dataset achiziționat și analizat (EDA realizată - vezi grafice).

    [x] Pipeline de preprocesare (Resize, Normalize) implementat în PyTorch.

    [x] Strategia de augmentare definită pentru a rezolva lipsa datelor.

    [x] Documentație actualizată.





