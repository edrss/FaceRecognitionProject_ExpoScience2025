import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import dlib
import bz2

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return f"{round(linear_val * 100, 2)}%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return f"{round(value, 2)}%"

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True

        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.compressed_path = "shape_predictor_68_face_landmarks_GTX.dat.bz2"

        # Extraire si le fichier .dat n’existe pas
        self.ensure_predictor()

        # Initialisation du détecteur et du prédicteur
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.encode_faces()

    def ensure_predictor(self):
        if not os.path.exists(self.predictor_path):
            if os.path.exists(self.compressed_path):
                print("Extraction du modèle .dat à partir du fichier .bz2...")
                with bz2.BZ2File(self.compressed_path, 'rb') as f_in:
                    with open(self.predictor_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                print("Extraction terminée.")
            else:
                print(f"Erreur : fichier compressé {self.compressed_path} introuvable.")
                sys.exit(1)

    def encode_faces(self):
        faces_dir = 'Faces'
        if not os.path.exists(faces_dir):
            print(f"Le dossier '{faces_dir}' n'existe pas.")
            sys.exit(1)

        for image_name in os.listdir(faces_dir):
            image_path = os.path.join(faces_dir, image_name)
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(os.path.splitext(image_name)[0])
            else:
                print(f"Aucun visage détecté dans l'image {image_name}.")

        print("Visages connus :", self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Impossible d'accéder à la caméra.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Échec de la capture de la vidéo.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if self.process_current_frame:
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Inconnu"
                    confidence = "0%"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Affichage du rectangle et nom
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Repères 68 points
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dlib_rect = dlib.rectangle(left, top, right, bottom)
                landmarks = self.predictor(gray, dlib_rect)

                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow('Reconnaissance Faciale + 68 points', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
