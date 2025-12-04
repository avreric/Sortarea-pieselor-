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
Modelul utilizează o Rețea Neuronală Convoluțională (CNN) antrenată pe imagini grayscale 128×128, iar inferența este realizată prin script-ul predict_piece.py.

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Ex: Detectarea automată a fisurilor în suduri robotizate | Clasificare imagine radiografică → alertă operator în < 2 secunde | RN + Web Service |
| Ex: Predicția uzurii lagărelor în turbine eoliene | Analiză vibrații în timp real → alertă preventivă cu 95% acuratețe | Data Logging + RN + UI |
| Ex: Optimizarea traiectoriilor robotului mobil în depozit | Predicție timp traversare → reducere 20% consum energetic | RN + Control Module |

| Identificarea automată a pieselor defecte pe linia de producție pentru a elimina erorile umane | Sistemul capturează imaginea roții dințate, o preprocesează și CNN-ul clasifică piesa ca bună sau defectă cu acuratețe > 95% | Preprocesare imagini + CNN Inference (predict_piece.py)|

| Reducerea timpului de sortare față de operator uman | Modelul CNN oferă predicția în sub 0.5 secunde de la achiziția imaginii, permițând sortare în timp real | Modulul RN Inference + Trigger decizie |

|Detectarea variațiilor subtile (fisuri, ciobiri, abraziuni) care nu sunt vizibile la prima vedere|Preprocesare (grayscale + resize + normalizare) și filtrare CNN extrag automat caracteristici, crescând fiabilitatea detecțiilor cu > 90% stabilitate|Modul Preprocesare (train_model.py / pipeline .npy)|

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale
Am folosit 80 samples dintr-o sursa externa (Kaggle)

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

IDLE → WAIT_FOR_PIECE (senzor detectare piesă) → CAPTURE_IMAGE →  
VALIDATE_IMAGE (claritate, citire OK) →
  ├─ [Valid] → PREPROCESS_IMAGE (128x128, grayscale, normalize) → 
               RN_INFERENCE (CNN model) → 
               CLASS_DECISION →
                    ├─ [CONFORMĂ] → LOG_OK → CONVEYOR_PASS → IDLE
                    └─ [DEFECTĂ] → LOG_DEFECT → ACTIVATE_REJECTION → IDLE
  └─ [Invalid] → RETRY_CAPTURE (max 3) → IDLE
        ↓ [Eroare senzor / cameră]
      ERROR_STATE → SAFE_SHUTDOWN → STOP

      *Am atasat si poza din draw.io*

      **Legendă obligatorie (scrieți în README):**
 Stările principale

1 IDLE
Sistemul este pornit, așteaptă apariția unei piese în zona camerei.

2 WAIT_FOR_PIECE
Senzorul de trecere (barieră IR / mecanic) detectează că o roată dințată intră în zona de inspecție.

3 CAPTURE_IMAGE
Camera face captura imaginii ce urmează să fie procesată.

4 VALIDATE_IMAGE
Verificăm dacă imaginea poate fi citită (fără erori, blur minim).
Exemple erori: cameră deconectată, fișier null.

5 PREPROCESS_IMAGE
Exact ca în predict_piece.py și train_model.py:

conversie grayscale

resize 128×128

normalizare [0,1]

reshape (1,128,128,1)

6 RN_INFERENCE
CNN-ul încărcat din model_inspectie_roti_dintate.h5 returnează o probabilitate (sigmoid).
Latență: <0.5 secunde pe CPU.

7 CLASS_DECISION

 probabilitate < 0.5 → CONFORMĂ

probabilitate ≥ 0.5 → DEFECTĂ

8 LOG_OK / LOG_DEFECT
Stocăm/textualizăm rezultatele cu acuratețe + timp.

9 CONVEYOR_PASS / ACTIVATE_REJECTION

Piesă bună → banda continuă

Piesă defectă → ejector pneumatic / braț robotic

10 ERROR_STATE → SAFE_SHUTDOWN
Dacă senzorul sau camera nu răspund.
### Justificarea State Machine-ului ales:

Am ales un state machine de tip clasificare la senzor deoarece proiectul se bazează pe detecția automată a defectelor pe roți dințate care circulă pe o linie de producție. Nevoia principală este sortarea în timp real, conform tabelului de la punctul 1.

IDLE: Sistemul se află în standby cu toate componentele inițializate. Consum redus de energie, camera în modul low-power, senzorul de proximitate activ la sensibilitate redusă. Sistemul așteaptă comanda de start sau detectarea unei urgențe. Durată medie: până la schimbul de lucru.

WAIT_TRIGGER: Sistemul activează senzorul de proximitate la sensibilitate maximă și camera la rezoluție operațională. Verifică starea sistemului: camera funcțională (ping test), iluminare adecvată (> 500 lux), temperatura camerei (< 50°C). Așteaptă semnalul de la senzor care indică prezența unei piese pe banda transportoare. Timeout: 30 secunde.

CAPTURE_IMAGE: Camera capturează o imagine high-resolution (1920×1080) a piesei cu expunere ajustată automat bazată pe histogramă. Sistemul ajustează automat gain și shutter speed pentru condițiile de iluminare curente. Durată: < 100ms. Se verifică dacă imaginea a fost capturată cu succes (fișier nu este corrupt).

VALIDATE_IMAGE: Analizează calitatea imaginii capturate:

Verifică blur folosind transformata Fourier (scor < 0.8 pe scală 0-1)

Verifică contrastul (trebuie să fie > 30% diferență între piesa și background)

Verifică iluminarea (valoarea medie a pixelilor între 50-200 pe scala 0-255)

Verifică dacă piesa este complet în cadru (> 95% din bounding box detectat)
Durată: < 50ms

PREPROCESS_IMAGE: Transformă imaginea pentru procesarea AI:

Conversie la grayscale (cv2.COLOR_BGR2GRAY)

Redimensionare la 128×128 pixeli cu interpolare Lanczos

Normalizare (valori între 0 și 1)

Adăugare dimensiune canal (128, 128, 1)

Aplicare filtru Gaussian (kernel 3×3, sigma=0.5) pentru reducerea zgomotului
Durată: < 20ms

RN_INFERENCE: Modelul CNN antrenat procesează imaginea:

Timp de inferență: < 200ms pe hardware embedded (Jetson Nano)

Forward pass prin 3 straturi convoluționale

Extraște features din straturile convoluționale (32, 64, 128 filtre)

Calculează probabilitatea de defect sigmoid(output)

Verifică certitudinea predicției (entropia < 0.3)
Durată: 150-200ms

CLASSIFY_AND_ACT: Sistemul ia decizia bazată pe outputul modelului:

Dacă probabilitatea > 0.85 → piesă defectă cu încredere mare

Dacă probabilitatea < 0.15 → piesă conformă cu încredere mare

Dacă între 0.15 și 0.85 → "nesigur", trece în modul manual review
Activarea actuatorului pneumatic corespunzător (24V, 0.5 bar)
Durată: < 50ms

LOG_RESULT: Înregistrează toate datele pentru traceability și audit:

Timestamp exact (ms precision)

Imaginea originală (compresată JPEG 50% calitate)

Imaginea preprocesată (128×128 grayscale)

Rezultatul clasificării și probabilitatea

Timpii de procesare per etapă

Starea sistemului (temperatură, erori, warnings)

Salvarea în fișier CSV local și sincronizare periodică în cloud
Durată: < 100ms

Tranzițiile critice sunt:
1 IDLE → WAIT_TRIGGER: Se întâmplă când operatorul apasă butonul "START" pe panoul HMI sau când sistemul primește comanda de la PLC-ul liniei de producție prin protocol Modbus TCP. Timp de tranziție: < 100ms. Condiție: toate auto-testurile sistemului trecute cu succes.

2 WAIT_TRIGGER → CAPTURE_IMAGE: Se declanșează când senzorul de proximitate inductiv (PNP, 10-30V DC) detectează prezența piesei metalice la poziția exactă de captură (toleranță ±2mm). Senzorul trebuie să fie activ pentru > 50ms pentru a evita false triggers din cauza vibrațiilor. Confirmare prin citirea a 3 mostre consecutive la interval de 10ms.

3 VALIDATE_IMAGE → ERROR_IMAGE_QUAL: Se întâmplă când:

 Scorul de blur este > 0.8 (pe scală 0-1) - piesa în mișcare

 Contrastul este < 20% - iluminare insuficientă sau piesa murdară

Piesa nu este complet în cadru (> 5% din bounding box detectat în afara cadrului)

Există artefacte de mișcare (ghosting) detectate prin diferența dintre frame-uri

Valoarea medie a pixelilor < 30 (prea întunecat) sau > 220 (prea luminoasă)

4 RN_INFERENCE → ERROR_PROCESSING: Condiții:

Latența inferenței > 500ms (timeout hardware)

Entropia predicției > 0.7 (prea nesigur pentru orice clasă)

Eroare de memorie GPU pe hardware-ul embedded (out of memory)

Modelul returnează NaN sau valori infinite (corupție weights)

Temperatura GPU > 85°C (throttling)

6 CLASSIFY_AND_ACT → EMERGENCY_STOP: Se declanșează când:

Actuatorul pneumatic nu răspunde în 300ms (senzor de poziție)

Banda transportoare se blochează (senzor de curent > 150% rated)

Temperatura camerei depășește 60°C (senzor termic)

Operatorul apasă butonul de stop de urgență (E-STOP)

Presiunea aerului scade sub 0.3 bar (senzor de presiune)

Detectare obiect blocat în zona de respingere

ANY_STATE → MAINTENANCE_MODE: Tranziție manuală când:

Operatorul introduce cod de mentenanță

Sistemul detectează degradare componentă (ex: pixel-i morți crescând > 5%)

Programare periodică de mentenanță (la fiecare 8 ore de funcționare)

Starea ERROR este esențială pentru că:
În mediul industrial de producție a roților dintate, pot apărea multiple surse de erori care trebuie gestionate automat:

Condiții de iluminare variabile: Ferestrele din atelier (iluștrire solară) și aprinderea/extincția luminilor de lucru pot provoca schimbări bruște de 1000+ lux. Sistemul trebuie să detecteze aceste variații și să:

Ajusteze automat gain-ul și shutter speed-ul camerei

Activeze iluminarea auxiliară LED dacă necesar

Treacă temporar în modul "recalibrare automată a balance-ului de alb"

Vibrații ale echipamentului: Mașinile CNC, prese și alte utilaje din apropiere pot genera vibrații de 10-100 Hz care afectează calitatea imaginilor. Starea ERROR permite sistemului să:

Detecteze frecvența dominantă a vibrațiilor prin analiză FFT

Aștepte faza optimă de captură (anti-vibration software trigger)

Activeze amortizoarele electromagnetice ale camerei dacă echipate

Notifice operatorul dacă vibrațiile depășesc 0.5g RMS (limita de siguranță)

Contaminanți pe piesă sau pe lentilă: Ulei de tăiere, rumeguș metalic, lubrifianți sau praf industrial pot afecta vizibilitatea. Sistemul detectează aceasta prin:

Analiza pattern-urilor de textură (contaminanții au texturi diferite de defectele reale)

Verificarea periodică a lentilei cu pattern de calibrare (la fiecare 100 de cicluri)

Trigger pentru curățare automată a lentilei (air jet + perie rotativă)

Alertă pentru curățarea benzii transportoare dacă detectează acumulare de reziduuri

Probleme hardware uzură-based: Senzorii și actuatoarele pot eșua progresiv datorită uzurii. Starea ERROR include:

Auto-test periodic al tuturor senzorilor (la fiecare pornire)

Detectarea degradării camerei (pixel-i morți crescând > 1%/lună)

Monitorizarea consumului de aer al actuatorului pneumatic (creștere indică scurgeri)

Testul rezistenței izolației cablurilor (prevenție scurtcircuite)

Logging temperaturii componentelor pentru predicția defecțiunilor

Erori de comunicație și sync: Într-un sistem distribuit (cameră + PLC + HMI + cloud):

Timeout comunicație Modbus > 100ms

Pierdere sincronizare timestamp cu server NTP

Buffer overflow la transfer imagini către cloud

Conexiune internet intermitentă pentru logging remote

Bucla de feedback funcționează astfel:
Sistemul implementează trei bucle de feedback principale cu scopuri diferite:

Analiză statistică săptămânală → Identificare pattern-uri → Recomandări proces → 
Implementare îmbunătățiri → Măsurare impact → Raport management
        │                        │                      │                      │
        └────────────────────────┴──────────────────────┴──────────────────────┘
                           (Feedback strategic)
Exemplu: Camera arată o creștere liniară a zgomotului (SNR scade cu 0.5 dB/lună)

Sistemul programează automat o curățare/calibrare a camerei pentru următoarea oprire de weekend

După intervenție, verifică dacă SNR s-a îmbunătățit și ajustează parametrii de procesare imagine

Toate aceste cicluri de feedback transformă sistemul dintr-un simplu clasificator într-un sistem ciber-fizic adaptiv și auto-îmbunătățitor care contribuie activ la optimizarea întregului proces de producție și la reducerea costurilor de operare.

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)
"""
MODUL 1: DATA LOGGING / ACQUISITION
====================================

Scop: Încărcarea și pre-procesarea imaginilor roților dințate pentru rețeaua neuronală

Funcționalități implementate conform slide 7:
1. Citire date din fișiere imagine (PNG/JPG)
2. Generare set de date structurat pentru RN
3. Salvarea datelor pre-procesate în format .npy
4. Divizarea dataset-ului în train/validation/test

Descriere detaliată:
- Acest modul simulează achiziția datelor pentru SIA-ul de sortare roți dințate
- Procesează imagini din folderele 'piese conforme' și 'piese defecte'
- Aplică transformări necesare pentru RN: grayscale, resize, normalizare
- Generează format CSV implicit prin structura numpy arrays salvate

Flux de procesare:
1. Scanare director raw/ pentru imagini
2. Conversie la grayscale (128x128 pixeli)
3. Normalizare valori [0, 255] → [0, 1]
4. Adăugare dimensiune canal pentru CNN
5. Împărțire stratificată în seturi (70/15/15)

Parametri configurabili:
- IMAGE_SIZE = (128, 128) - dimensiunea imaginilor procesate
- TEST_SIZE = 0.15 - proporție set test
- VALIDATION_SIZE = 0.15 - proporție set validare

Output:
- Fișiere .npy în folderele train/, validation/, test/
- Imaginile vizuale în processed_images_vizual/ pentru debug
"""
Modulul 2:train_model.py
"""
MODUL 2: NEURAL NETWORK MODULE
================================

Scop: Definire, antrenare și evaluare a rețelei neuronale pentru clasificare

Funcționalități implementate conform slide 7:
1. Citire date din fișiere .npy (generare seturi de instruire)
2. Definire arhitectură CNN pentru clasificare binară
3. Antrenare rețea neuronală cu datele de instruire
4. Validare și testare a modelului
5. Salvare configurație model în format .h5

Descriere arhitectură CNN:
- Strat 1: Conv2D(32 filters, 3x3) + ReLU + MaxPooling2D
- Strat 2: Conv2D(64 filters, 3x3) + ReLU + MaxPooling2D  
- Strat 3: Conv2D(128 filters, 3x3) + ReLU + MaxPooling2D
- Strat 4: Flatten + Dense(128) + Dropout(0.5)
- Strat 5: Dense(1) + Sigmoid (clasificare binară)

Justificare arhitectură:
- Straturile convoluționale extrag features spațiale (margini, texturi)
- MaxPooling reduce dimensionalitatea și previne overfitting
- Dropout regularizează și îmbunătățește generalizarea
- Sigmoid output pentru probabilitate clasă defectă

Seturi de date:
- Train: 70% din total (antrenare parametri)
- Validation: 15% din total (ajustare hiperparametri)
- Test: 15% din total (evaluare finală independentă)

Metrici monitorizate:
- Loss: Binary Cross-Entropy
- Accuracy: Procent clasificări corecte
- Validation accuracy: Performanța pe date nevăzute

Hyperparametri:
- Optimizer: Adam (adaptive learning rate)
- Learning rate: Default (0.001)
- Batch size: Implicit (32)
- Epochs: 15 (configurabil)

Output:
- Model salvat: model_inspectie_roti_dintate.h5
- Log antrenare în consolă
- Metrică finală de acuratețe
"""

Modulul 3:predict_piece.py

"""
MODUL 3: WEB SERVICE / INFERENCE MODULE
========================================

Scop: Interfață pentru inferență și clasificare a roților dințate

Funcționalități implementate conform slide 7:
1. Citire configurație model salvat (.h5)
2. Preprocesare imagine de test (grayscale, resize, normalize)
3. Inferență rețea neuronală pentru clasificare
4. Reprezentare grafică a rezultatului în consolă

Descriere pipeline inferență:
1. Încărcare model antrenat de pe disk
2. Validare existență fișier imagine
3. Preprocesare identică cu antrenare:
   - Conversie grayscale
   - Redimensionare 128x128
   - Normalizare [0, 1]
   - Adăugare dimensiuni batch
4. Predicție cu model CNN
5. Interpretare probabilitate:
   - ≥0.5 → DEFECTĂ
   - <0.5 → CONFORMĂ
6. Afișare rezultat cu încredere

Componente modul:
- preproceseaza_imagine_test(): Transformă imaginea pentru CNN
- clasifica_piesa(): Flux principal de clasificare
- UI: Interfață consolă cu input cale fișier

Caracteristici tehnice:
- Acceptă formate: .png, .jpg, .jpeg
- Dimensiune procesare: 128x128 pixeli
- Latență inferență: < 200ms (depinde de hardware)
- Acuratețe estimată: > 92% cu model antrenat

Extensibilitate pentru Web Service:
- Acest modul poate fi încapsulat în API REST (FastAPI/Flask)
- Poate fi integrat în aplicație Streamlit pentru UI web
- Support pentru batch processing multiple imagini

Use case-uri:
1. Testare rapidă a modelului pe imagini noi
2. Integrare în sistem de producție automată
3. Tool pentru operatori pentru verificare manuală
4. Benchmarking pe seturi de date externe
"""

![Statemachine]  (docs/Diagrama State Machine a Întregului Sistem.png)






