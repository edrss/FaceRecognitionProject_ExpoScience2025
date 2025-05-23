## Projet de Reconnaissance Faciale en Temps Réel

Ce guide explique pas à pas la mise en place d'une application de reconnaissance faciale en temps réel avec Python, OpenCV et dlib, incluant la détection et l'affichage des 68 points de repérage facial.

---

### 📋 Prérequis

* **Python 3.8 à 3.10** (3.11+ non supporté par certaines dépendances)
* **pip** (gestionnaire de paquets Python)
* **Git** (pour récupérer certains modèles)
* **Visual Studio Community 2022** (Windows) ou outils de compilation C++ / build-essential (Linux/Mac)
* **7-Zip** ou équivalent pour décompresser manuellement, si besoin

---

### 📦 Librairies Python et modèles

Liste des dépendances à installer :

```bash
pip install --upgrade pip setuptools wheel
pip install opencv-python
pip install face_recognition
pip install cmake
# Installation de dlib via wheel précompilé (consulter le README de dlib-wheels)
# Exemple :
pip install dlib-19.24.2-cp310-cp310-win_amd64.whl
```

Modèle de repérage facial 68 points (dlib) :

* Télécharger le fichier `shape_predictor_68_face_landmarks.dat.bz2` depuis le dépôt officiel de dlib-models
* Placer le fichier compressé à la racine du projet

---

### 📁 Structure du projet

```yaml
project_root: "C:\Users\cjean\Documents\Face Recognition Project - ExpoScience 2025"
contents:
  .idea/: "configuration PyCharm"
  .venv/: "environnement virtuel Python"
  Faces/: "dossier d’images de référence (JPEG/PNG)"
  haarcascade_frontalface_default.xml: "modèle de détection OpenCV (optionnel)"
  main.py: "script principal de reconnaissance faciale"
  README.md: "documentation du projet"
  requirements.txt: "liste des dépendances Python"
  shape_predictor_68_face_landmarks.dat: "modèle dlib 68 points (extrait)"
  shape_predictor_68_face_landmarks_GTX.dat.bz2: "archive compressée du modèle dlib"
```

---

### 🚀 Étapes d'installation et de configuration

1. **Cloner ou télécharger le projet**

   ```bash
   git clone https://github.com/votre-utilisateur/Face_Recognition_Project.git
   cd Face_Recognition_Project
   ```

2. **Créer et activer un environnement virtuel** (fortement recommandé) :

   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Mettre à jour pip et installer les dépendances** :

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

   *Le fichier `requirements.txt` doit contenir :*

   ```text
   opencv-python
   face_recognition
   cmake
   # dlib (installé séparément si nécessaire)
   ```

4. **Installer dlib** :

   * Si la compilation locale échoue, télécharger la wheel précompilée adaptée à votre version de Python et exécuter :

     ```bash
     pip install chemin/vers/dlib-*.whl
     ```

5. **Extraire le modèle dlib 68 points** :

   ```python
   import bz2
   with bz2.BZ2File('shape_predictor_68_face_landmarks_GTX.dat.bz2', 'rb') as f_in:
       with open('shape_predictor_68_face_landmarks.dat', 'wb') as f_out:
           f_out.write(f_in.read())
   ```

---

### 🎬 Utilisation

Lancez le script Python :

```bash
python script.py
```

* Appuyez sur `q` pour quitter la fenêtre.
* Le script :

  1. Charge et encode les visages connus dans `Faces/`
  2. Ouvre la webcam et détecte les visages en temps réel
  3. Compare chaque visage détecté aux visages de référence
  4. Affiche un cadre rouge, le nom et le taux de confiance
  5. Extrait et dessine les 68 points de repérage facial en vert

---

### 🔗 Ressources & Références

* [Face Detection using Python and OpenCV with webcam (GeeksforGeeks)](https://www.geeksforgeeks.org/face-detection-using-python-and-opencv-with-webcam/)
* [Face Detection & Recognition avec OpenCV et Python (TechVidvan)](https://techvidvan.com/tutorials/face-detection-recognition-opencv-python/?amp=1)
* [Building Real-Time Face Recognition with Python (Medium)](https://medium.com/@suditi/building-real-time-face-recognition-with-python-b0584900d631)
* [Tutoriel vidéo : Live Face Detection with Python & OpenCV (YouTube)](https://www.youtube.com/watch?v=tl2eEBFEHqM)
* [Modèles dlib 68 points (GitHub)](https://github.com/davisking/dlib-models)
