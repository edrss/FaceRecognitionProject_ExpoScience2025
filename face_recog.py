import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import dlib
import bz2
import pickle
from datetime import datetime

def face_confidence(face_distance, face_match_threshold=0.6):
    """
    Convertit une distance de visage en pourcentage de confiance.
    """
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return f"{round(linear_val * 100, 2)}%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return f"{round(value, 2)}%"

class FaceRecognition:
    """
    Module de reconnaissance faciale basé sur face_recognition et dlib.
    - encode_faces() : encode tous les visages présents dans le dossier 'Faces/'.
    - load_encodings(path) : recharge un fichier d'encodages (.npz ou .pkl).
    - save_encodings(path) : sauvegarde les encodages et noms connus.
    - add_face(name, frame_bgr) : ajoute un nouveau visage (image BGR) dans 'Faces/' + encodage.
    - process_frame(frame_bgr, tolerance) : renvoie (frame_annoté, [noms_chiffrés]) pour Streamlit.
    """

    def __init__(self,
                 faces_dir: str = "Faces",
                 predictor_bz2: str = "shape_predictor_68_face_landmarks_GTX.dat.bz2",
                 predictor_dat: str = "shape_predictor_68_face_landmarks.dat",
                 encodings_file: str = "encodings/known_faces.npz"):
        """
        - faces_dir       : dossier où sont stockées toutes les images de visages connues (.jpg/.png)
        - predictor_bz2   : chemin vers le fichier .bz2 compressé du modèle de landmarks Dlib
        - predictor_dat   : chemin vers le fichier .dat extrait
        - encodings_file  : fichier .npz (ou .pkl) où l’on sauvegarde les encodages et noms
        """
        self.faces_dir       = faces_dir
        self.predictor_path  = predictor_dat
        self.compressed_path = predictor_bz2
        self.encodings_file  = encodings_file

        # ------------------------------------------------
        # 1. S’assurer que le dossier de visages existe
        # ------------------------------------------------
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        # ------------------------------------------------
        # 2. S’assurer que le modèle Dlib est extrait
        # ------------------------------------------------
        self.ensure_predictor()

        # ------------------------------------------------
        # 3. Charger le détecteur et le prédicteur Dlib
        # ------------------------------------------------
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # ------------------------------------------------
        # 4. Initialiser les listes d’encodages et de noms
        # ------------------------------------------------
        self.known_face_encodings = []  # liste d’array (128d)
        self.known_face_names     = []  # liste de chaînes

        # ------------------------------------------------
        # 5. Charger les encodages existants si fichier présent
        # ------------------------------------------------
        if os.path.exists(self.encodings_file):
            try:
                self.load_encodings(self.encodings_file)
                print(f"[INFO] Encodages chargés depuis '{self.encodings_file}'.")
            except Exception as e:
                print(f"[WARNING] Impossible de charger '{self.encodings_file}' ({e}). Recréation à partir de '{self.faces_dir}/'.")
                self.encode_faces()
                self.save_encodings(self.encodings_file)
        else:
            # Si pas d’encodages, on encode tout le dossier 'Faces/'
            self.encode_faces()
            # Et on sauvegarde immédiatement
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            self.save_encodings(self.encodings_file)

        # Contrôle pour alterner le traitement deux frames sur quatre (gain performance)
        self.process_current_frame = True

    # ────────────────────────────────────────────────────────────
    # Méthode pour s’assurer que le modèle Dlib est bien
    # extrait depuis le .bz2 si nécessaire
    # ────────────────────────────────────────────────────────────
    def ensure_predictor(self):
        """
        Si le fichier .dat n’existe pas, tente d’extraire depuis le .bz2.
        """
        if not os.path.exists(self.predictor_path):
            if os.path.exists(self.compressed_path):
                print("Extraction du modèle .dat à partir du fichier .bz2…")
                with bz2.BZ2File(self.compressed_path, 'rb') as f_in:
                    with open(self.predictor_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                print("Extraction terminée.")
            else:
                print(f"[ERROR] Le fichier compressé '{self.compressed_path}' est introuvable.")
                sys.exit(1)

    # ────────────────────────────────────────────────────────────
    # Méthode 1 : encoder tous les visages stockés dans 'Faces/'
    # ────────────────────────────────────────────────────────────
    def encode_faces(self):
        """
        Scanne le dossier 'Faces/' et pour chaque image valide (.jpg/.png),
        calcule l’encodage du visage (avec face_recognition) et ajoute
        à self.known_face_encodings + self.known_face_names (nom du fichier sans extension).
        """
        self.known_face_encodings = []
        self.known_face_names     = []

        if not os.path.exists(self.faces_dir):
            print(f"[ERROR] Le dossier '{self.faces_dir}' n'existe pas.")
            sys.exit(1)

        for image_name in os.listdir(self.faces_dir):
            image_path = os.path.join(self.faces_dir, image_name)
            # Ne traiter que fichiers JPG/JPEG/PNG
            ext = image_name.lower().split('.')[-1]
            if ext not in ("jpg", "jpeg", "png"):
                continue

            # Charger l’image via face_recognition (RGB)
            face_image = face_recognition.load_image_file(image_path)
            encodings  = face_recognition.face_encodings(face_image)
            if len(encodings) > 0:
                # On prend le premier encodage trouvé (suppose 1 visage par image)
                self.known_face_encodings.append(encodings[0])
                # Le nom associé : nom du fichier sans extension
                nom_cle = os.path.splitext(image_name)[0]
                self.known_face_names.append(nom_cle)
            else:
                print(f"[WARNING] Aucun visage détecté dans '{image_name}'. Ignoré.")

        print(f"[INFO] Visages connus : {self.known_face_names}")

    # ────────────────────────────────────────────────────────────
    # Méthode 2 : sauvegarder les encodages + noms dans un .npz
    # ────────────────────────────────────────────────────────────
    def save_encodings(self, path: str):
        """
        Sauvegarde self.known_face_encodings et self.known_face_names dans un fichier .npz.
        On convertit known_face_encodings en matrice numpy (n,128) pour la sauvegarde.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convertir la liste d’encodages en numpy array de forme (N,128)
        if len(self.known_face_encodings) > 0:
            enc_array = np.vstack(self.known_face_encodings)
        else:
            enc_array = np.empty((0, 128), dtype=np.float64)

        # Convertit la liste de noms en array d’objets byte-string
        names_array = np.array(self.known_face_names, dtype=object)

        # Sauvegarde compressée
        np.savez_compressed(path, encodings=enc_array, names=names_array)
        print(f"[INFO] Encodages enregistrés dans '{path}' ({enc_array.shape[0]} visages).")

    # ────────────────────────────────────────────────────────────
    # Méthode 3 : charger les encodages + noms depuis un .npz
    # ────────────────────────────────────────────────────────────
    def load_encodings(self, path: str):
        """
        Recharge un fichier .npz généré par save_encodings().
        Modifie self.known_face_encodings et self.known_face_names en conséquence.
        """
        loader = np.load(path, allow_pickle=True)
        enc_array   = loader["encodings"]   # forme (N,128)
        names_array = loader["names"]       # array d’objets (N,)

        self.known_face_encodings = [enc_array[i] for i in range(enc_array.shape[0])]
        self.known_face_names     = [str(names_array[i]) for i in range(len(names_array))]
        print(f"[INFO] {len(self.known_face_names)} encodages chargés depuis '{path}'.")

    # ────────────────────────────────────────────────────────────
    # Méthode 4 : ajouter un visage (image BGR) à la base
    # ────────────────────────────────────────────────────────────
    def add_face(self, name: str, frame_bgr: np.ndarray):
        """
        Ajoute un nouveau visage à la base :
         1) Sauvegarde l’image BGR sous 'Faces/{name}_{timestamp}.jpg'
         2) Calcule l’encodage et l’ajoute à self.known_face_encodings et self.known_face_names
        Retourne le chemin du fichier sauvegardé (string).
        """
        # Créeer un nom de fichier unique
        ts = int(datetime.now().timestamp())
        filename = f"{name}_{ts}.jpg"
        save_path = os.path.join(self.faces_dir, filename)

        # Sauvegarde l’image BGR (OpenCV)
        cv2.imwrite(save_path, frame_bgr)

        # Calcule l’encodage sur l’image RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if len(encs) == 0:
            print(f"[WARNING] Aucun visage détecté dans l’image capturée pour '{name}'.")
            # Supprime le fichier créé car pas de visage
            os.remove(save_path)
            return None

        # On prend le premier encodage
        encoding = encs[0]
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        print(f"[INFO] Nouveau visage '{name}' ajouté (fichier : {filename}).")
        return save_path

    # ────────────────────────────────────────────────────────────
    # Méthode 5 : application en temps réel d’une frame BGR
    # (Annotation + détection de noms)
    # ────────────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray, tolerance: float = 0.6):
        """
        Prend une image OpenCV (BGR), détecte et reconnaît les visages connus,
        dessine rectangles, noms + pourcentage, points landmarks(68).
        Renvoie :
         - annotated_frame (BGR) : l’image annotée
         - detected_names   (list of str) : liste des noms détectés (***sans*** le pourcentage)
        """
        # 1) On redimensionne ×0.25 pour accélérer
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        detected_names = []

        if self.process_current_frame:
            # 2a) Détection des visages + encodages (sur la petite frame)
            self.face_locations = face_recognition.face_locations(rgb_small)
            self.face_encodings = face_recognition.face_encodings(rgb_small, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance)
                name = "Inconnu"
                confidence = "0%"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index], face_match_threshold=tolerance)

                self.face_names.append(f"{name} ({confidence})")
                detected_names.append(name)

        # Toggle pour ne traiter qu’une frame sur deux
        self.process_current_frame = not self.process_current_frame

        # 3) Pour chaque visage détecté, on dessine dans l’image d’origine (×4)
        for (top, right, bottom, left), name_text in zip(self.face_locations, self.face_names):
            top    *= 4
            right  *= 4
            bottom *= 4
            left   *= 4

            # Rectangle rouge + bande de texte
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame_bgr, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame_bgr, name_text, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Landmarks (68 points) via Dlib
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            landmarks = self.predictor(gray, dlib_rect)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)

        # 4) Retourne l’image annotée (BGR) et la liste de noms détectés (juste la clé)
        return frame_bgr, detected_names

    # ──────────────────────────────────────────────────────────────────────────
    # Méthode facultative (usage console) pour afficher la webcam OpenCV
    # ──────────────────────────────────────────────────────────────────────────
    def run_recognition(self):
        """
        Lecture en boucle de la webcam, annotation en OpenCV, quitte à 'q'.
        Strictement pour usage console/debogage.
        """
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit("Impossible d'accéder à la caméra.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Échec de la capture vidéo.")
                break

            annotated_frame, _ = self.process_frame(frame)
            cv2.imshow('Reconnaissance Faciale + 68 points', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
