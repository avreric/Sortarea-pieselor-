# Documentație Set de Date - Inspecție Industrială

## Descriere Generală
Setul de date conține imagini top-down cu piese mecanice metalice (piese turnate). Scopul este clasificarea binară a acestora pentru controlul calității automatizat.

## Structura Claselor
1. **OK (Piese Bune):** Piese care nu prezintă defecte vizibile de turnare.
2. **Defective (Piese Defecte):** Piese care prezintă fisuri, pori, zgârieturi sau neregularități de formă.

## Specificații Tehnice
* **Format:** JPG/PNG
* **Canale:** 1 (Grayscale) convertit la 3 (RGB) pentru compatibilitate ViT.
* **Rezoluție originală:** 300x300 px (medie).
* **Preprocesare:** Redimensionare la 224x224 px.
