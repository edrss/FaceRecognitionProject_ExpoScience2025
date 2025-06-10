# streamlit_app.py

import os
import json
import time
import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
from pandas import DataFrame

from face_recog import FaceRecognition

# ----------------------------------------------------
# CONSTANTES CHEMINS & FICHIERS
# ----------------------------------------------------
PERSON_INFO_FILE = "person_info.json"
FACES_DIR        = "Faces"
ENCODINGS_DIR    = "encodings"
ENCODINGS_FILE   = os.path.join(ENCODINGS_DIR, "known_faces.npz")

# S’assurer que les dossiers existent
for d in (FACES_DIR, ENCODINGS_DIR):
    if not os.path.exists(d):
        os.makedirs(d)

# ----------------------------------------------------
# FONCTIONS DE GESTION DU JSON
# ----------------------------------------------------
def load_person_info():
    """Charge person_info.json ou renvoie {} si absent/corrompu."""
    try:
        with open(PERSON_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_person_info(d):
    """Sauvegarde le dict dans person_info.json (indenté)."""
    with open(PERSON_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------
# 1. CHARGEMENT DES FICHES & INITIALISATION
# ----------------------------------------------------
person_info = load_person_info()
# Format interne attendu :
# {
#   "identifiant_unique": {
#       "nom": "Dupont",
#       "prenom": "Pierre",
#       "nationalite": "Française",
#       "date_naissance": "1990-05-12",
#       "sexe": "Homme",
#       "occupation": "Étudiant",
#       "email": "pierre.dupont@mail.com",
#       "commentaires": "…",
#       "photo": "Faces/identifiant_timestamp.jpg"
#   },
#   ...
# }

fr = FaceRecognition(
    faces_dir=FACES_DIR,
    predictor_bz2="shape_predictor_68_face_landmarks_GTX.dat.bz2",
    predictor_dat="shape_predictor_68_face_landmarks.dat",
    encodings_file=ENCODINGS_FILE
)

# ----------------------------------------------------
# 2. CONFIGURATION STREAMLIT
# ----------------------------------------------------
st.set_page_config(page_title="Reconnaissance Faciale + Fiches", layout="wide")
st.title("🕵️‍♂️ ExpoScience2025 – Reconnaissance Faciale")

# ----------------------------------------------------
# 3. VARIABLES D’ÉTAT STREAMLIT
# ----------------------------------------------------
if "historique" not in st.session_state:
    st.session_state.historique = []            # [(timestamp, identifiant)]
if "dernier_identifiant" not in st.session_state:
    st.session_state.dernier_identifiant = None
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()
if "seen_ids_list" not in st.session_state:
    st.session_state.seen_ids_list = []
if "fiches_history" not in st.session_state:
    st.session_state.fiches_history = []

# Vars pour la capture multiple
if "capture_active" not in st.session_state:
    st.session_state.capture_active = False
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

# ----------------------------------------------------
# 4. BARRE LATÉRALE : Gestion des fiches
# ----------------------------------------------------
st.sidebar.header("⚙️ Gestion des fiches avancée")

# --- Section A : Capturer plusieurs images (pour l’encodage) ---
st.sidebar.subheader("➕ Capturer plusieurs images pour un même identifiant")
if st.sidebar.button("📷 Ouvrir caméra capture", key="btn_open_cam"):
    st.session_state.capture_active = True
if st.sidebar.button("✖️ Fermer caméra capture", key="btn_close_cam"):
    st.session_state.capture_active = False
    st.session_state.captured_images = []

cam_placeholder = st.sidebar.empty()
if st.session_state.capture_active:
    cap_live = cv2.VideoCapture(0)
    if not cap_live.isOpened():
        st.sidebar.error("Impossible d’accéder à la webcam.")
        st.session_state.capture_active = False
    else:
        ret, frame_live = cap_live.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, use_container_width=True)

            if st.sidebar.button("🔍 Capturer image", key="btn_capture_multiple"):
                st.session_state.captured_images.append(frame_live.copy())
                st.sidebar.success(f"{len(st.session_state.captured_images)} image(s) capturée(s).")
        else:
            st.sidebar.warning("Impossible de lire la webcam.")
        cap_live.release()
else:
    cam_placeholder.empty()

# Si on a capturé au moins une image, on affiche les vignettes
if st.session_state.captured_images:
    st.sidebar.markdown("**🖼️ Aperçu des images capturées :**")
    cols = st.sidebar.columns(3)
    for idx, img_bgr in enumerate(st.session_state.captured_images):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cols[idx % 3].image(img_rgb, use_container_width=True)
    st.sidebar.markdown("---")

    # Formulaire d’ajout (avec identifiant séparé de nom)
    st.sidebar.markdown("**📋 Remplir la fiche :**")
    identifiant    = st.sidebar.text_input("Identifiant unique (clé)", key="identifiant")
    nom_personne   = st.sidebar.text_input("Nom", key="nom")
    prenom         = st.sidebar.text_input("Prénom", key="prenom")
    nationalite    = st.sidebar.text_input("Nationalité", key="nationalite")
    date_naiss     = st.sidebar.date_input("Date de naissance", key="date_naiss", min_value=datetime(1900, 1, 1), max_value=datetime.now())
    sexe           = st.sidebar.selectbox("Sexe", ["Homme", "Femme", "Autre"], key="sexe")
    occupation     = st.sidebar.text_input("Occupation", key="occupation")
    email          = st.sidebar.text_input("Email", key="email")
    commentaires   = st.sidebar.text_area("Commentaires", key="commentaires")

    if st.sidebar.button("✅ Créer la fiche complète", key="btn_create_full"):
        # Vérifications
        if not identifiant:
            st.sidebar.error("Le champ « Identifiant » est requis.")
        elif identifiant in person_info:
            st.sidebar.error("Cet identifiant existe déjà.")
        elif not st.session_state.captured_images:
            st.sidebar.error("Capturez au moins une image.")
        else:
            # 1) On enregistre **toutes** les images pour l’encodage
            saved_paths = []
            for img_bgr in st.session_state.captured_images:
                sp = fr.add_face(identifiant, img_bgr)
                if sp:
                    saved_paths.append(sp)
            if not saved_paths:
                st.sidebar.error("Aucun visage valide dans les images.")
            else:
                # 2) On ne garde QU’UNE SEULE photo dans le JSON (la première)
                fiche_dict = {
                    "nom": nom_personne,
                    "prenom": prenom,
                    "nationalite": nationalite,
                    "date_naissance": date_naiss.strftime("%Y-%m-%d"),
                    "sexe": sexe,
                    "occupation": occupation,
                    "email": email,
                    "commentaires": commentaires,
                    "photo": saved_paths[0]  # UNE SEULE photo de profil
                }
                person_info[identifiant] = fiche_dict
                save_person_info(person_info)

                # 3) Sauvegarder les encodages
                fr.save_encodings(ENCODINGS_FILE)

                st.sidebar.success(f"✔️ Fiche « {identifiant} » créée (photo : {os.path.basename(saved_paths[0])}).")
                st.session_state.captured_images = []
                st.session_state.capture_active = False
                st.experimental_rerun()

# --- Section B : Modifier / Supprimer une fiche existante ---
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Modifier / Supprimer une fiche")
liste_ids = sorted(person_info.keys())
selected_id = st.sidebar.selectbox("Sélectionner un identifiant", [""] + liste_ids, key="select_existing")

if selected_id:
    data_sel = person_info[selected_id]
    with st.sidebar.expander(f"✏️ Fiche « {selected_id} »", expanded=True):
        # Affichage de la photo unique (champ "photo")
        photo_path = data_sel.get("photo", "")
        if photo_path and os.path.exists(photo_path):
            st.image(Image.open(photo_path), use_container_width=True)
        else:
            st.write("_Pas de photo._")

        st.markdown(f"**Identifiant (clé) :** `{selected_id}`")
        new_nom         = st.text_input("Nom", value=data_sel.get("nom", ""), key="modif_nom")
        new_prenom      = st.text_input("Prénom", value=data_sel.get("prenom", ""), key="modif_prenom")
        new_nationalite = st.text_input("Nationalité", value=data_sel.get("nationalite", ""), key="modif_nationalite")
        try:
            default_date = datetime.strptime(data_sel.get("date_naissance", "2000-01-01"), "%Y-%m-%d", min_value=datetime(1900, 1, 1), max_value=datetime.now())
        except:
            default_date = datetime(2000, 1, 1)
        new_date_naiss  = st.date_input("Date de naissance", value=default_date, key="modif_date_naiss")
        new_sexe        = st.selectbox(
            "Sexe",
            ["Homme", "Femme", "Autre"],
            index=["Homme","Femme","Autre"].index(data_sel.get("sexe", "Homme")),
            key="modif_sexe"
        )
        new_occupation  = st.text_input("Occupation", value=data_sel.get("occupation", ""), key="modif_occupation")
        new_email       = st.text_input("Email", value=data_sel.get("email", ""), key="modif_email")
        new_commentaires= st.text_area("Commentaires", value=data_sel.get("commentaires", ""), key="modif_commentaires")

        # Capturer une nouvelle photo pour remplacer l’ancienne :
        if st.button("📷 Capturer nouvelle photo", key="btn_capture_replace"):
            st.session_state.capture_active = True

        if st.session_state.capture_active:
            cap2 = cv2.VideoCapture(0)
            if not cap2.isOpened():
                st.sidebar.error("Impossible d’accéder à la webcam.")
                st.session_state.capture_active = False
            else:
                ret2, frame2 = cap2.read()
                if ret2:
                    st.sidebar.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if st.sidebar.button("🔍 Capturer pour remplacer", key="btn_replace_photo"):
                        # On ajoute l’encodage (sans changer l’ID), puis on remplace la photo stockée
                        sp_new = fr.add_face(selected_id, frame2)
                        if sp_new:
                            ancienne = data_sel.get("photo", "")
                            if ancienne and os.path.exists(ancienne):
                                os.remove(ancienne)
                            data_sel["photo"] = sp_new
                            save_person_info(person_info)
                            fr.save_encodings(ENCODINGS_FILE)
                            st.sidebar.success("✔️ Photo remplacée !")
                            st.session_state.capture_active = False
                            st.experimental_rerun()
                        else:
                            st.sidebar.error("Aucun visage détecté sur l’image.")
                cap2.release()

        col_mod1, col_mod2 = st.columns([1, 1])
        with col_mod1:
            if st.button("💾 Enregistrer modifications", key="btn_save_modif"):
                # L’identifiant ne bouge pas
                person_info[selected_id] = {
                    "nom": new_nom,
                    "prenom": new_prenom,
                    "nationalite": new_nationalite,
                    "date_naissance": new_date_naiss.strftime("%Y-%m-%d"),
                    "sexe": new_sexe,
                    "occupation": new_occupation,
                    "email": new_email,
                    "commentaires": new_commentaires,
                    "photo": data_sel.get("photo", "")  # on garde la photo existante
                }
                save_person_info(person_info)
                st.sidebar.success("✔️ Fiche modifiée.")
                st.experimental_rerun()

        with col_mod2:
            if st.button("❌ Supprimer cette fiche", key="btn_delete"):
                # Supprimer la photo du disque
                ancienne = data_sel.get("photo", "")
                if ancienne and os.path.exists(ancienne):
                    os.remove(ancienne)
                del person_info[selected_id]
                save_person_info(person_info)
                st.sidebar.success(f"🗑️ Fiche « {selected_id} » supprimée.")
                st.experimental_rerun()

# --- Section C : Réinitialiser historique & Tolérance ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Paramètres de détection")
tolerance = st.sidebar.slider("Tolérance (0.0–1.0)", min_value=0.4, max_value=0.8, value=0.6, step=0.01)
if st.sidebar.button("♻️ Réinitialiser historique"):
    st.session_state.historique = []
    st.session_state.dernier_identifiant = None
    st.session_state.seen_ids = set()
    st.session_state.seen_ids_list = []
    st.session_state.fiches_history = []
    st.session_state.webcam_active = False

# ----------------------------------------------------
# 5. MISE EN PAGE PRINCIPALE (deux colonnes)
# ----------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    mode = st.radio("Source d’image", ("↪️ Webcam", "📁 Importer une image"), horizontal=True)

    # ------ Mode Webcam ------
    if mode == "↪️ Webcam":
        col_start, col_stop = st.columns([1, 1])
        with col_start:
            if st.button("▶️ Démarrer la webcam", key="btn_start_cam"):
                st.session_state.webcam_active = True
        with col_stop:
            if st.button("⏹️ Arrêter la webcam", key="btn_stop_cam"):
                st.session_state.webcam_active = False

        video_placeholder = col1.empty()

        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Impossible d’accéder à la webcam.")
                st.session_state.webcam_active = False
            else:
                try:
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Lecture webcam impossible.")
                            break

                        # Reconnaissance faciale
                        annotated_frame, detected_ids = fr.process_frame(frame, tolerance=tolerance)

                        # Affichage BGR→RGB
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if detected_ids:
                            for ident in detected_ids:
                                # a) Historique si nouvel identifiant
                                if ident != st.session_state.dernier_identifiant:
                                    st.session_state.historique.append((timestamp, ident))
                                    st.session_state.dernier_identifiant = ident

                                # b) Si jamais vu cette session, afficher sa fiche (1 photo)
                                if ident not in st.session_state.seen_ids:
                                    st.session_state.seen_ids.add(ident)
                                    st.session_state.seen_ids_list.append(ident)
                                    st.session_state.fiches_history.append((timestamp, ident))

                                    with col2:
                                        with st.expander(f"▶️ Fiche de {ident}", expanded=True):
                                            if ident in person_info:
                                                info = person_info[ident]

                                                # Affichage de la photo unique
                                                photo_path = info.get("photo", "")
                                                if photo_path and os.path.exists(photo_path):
                                                    st.image(Image.open(photo_path), use_container_width=True)
                                                else:
                                                    st.write("_Pas de photo._")

                                                # Affichage des champs texte
                                                st.markdown(f"**Nom :** {info.get('nom','')}")
                                                st.markdown(f"**Prénom :** {info.get('prenom','')}")
                                                st.markdown(f"**Nationalité :** {info.get('nationalite','')}")
                                                st.markdown(f"**Date de naissance :** {info.get('date_naissance','')}")
                                                st.markdown(f"**Sexe :** {info.get('sexe','')}")
                                                st.markdown(f"**Occupation :** {info.get('occupation','')}")
                                                st.markdown(f"**Email :** {info.get('email','')}")
                                                st.markdown(f"**Commentaires :** {info.get('commentaires','')}")

                                                # Bouton téléchargement JSON
                                                fiche_json = json.dumps(info, ensure_ascii=False, indent=2)
                                                st.download_button(
                                                    label="💾 Télécharger la fiche JSON",
                                                    data=fiche_json,
                                                    file_name=f"{ident}.json",
                                                    mime="application/json"
                                                )
                                            else:
                                                st.write("❗ Aucune fiche disponible pour cet identifiant.")
                        else:
                            vide = "Aucun visage détecté"
                            if vide != st.session_state.dernier_identifiant:
                                st.session_state.historique.append((timestamp, vide))
                                st.session_state.dernier_identifiant = vide

                        time.sleep(0.03)
                finally:
                    cap.release()

    # ------ Mode Import d’image ------
    else:
        uploaded_file = st.file_uploader("Téléverse une image (.jpg/.png)", type=["jpg","jpeg","png"])
        if uploaded_file is not None:
            try:
                img_pil = Image.open(uploaded_file)
                img_pil = ImageOps.exif_transpose(img_pil)
                image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except:
                st.error("Impossible de décoder l’image.")
                image = None

            if image is not None:
                try:
                    annotated, detected_ids = fr.process_frame(image, tolerance=tolerance)
                except Exception as e:
                    st.error(f"Erreur pendant le traitement : {e}")
                    annotated, detected_ids = image.copy(), []

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                col1.image(annotated_rgb, caption="Image annotée", use_container_width=True)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if detected_ids:
                    for ident in detected_ids:
                        if ident != st.session_state.dernier_identifiant:
                            st.session_state.historique.append((timestamp, ident))
                            st.session_state.dernier_identifiant = ident
                        if ident not in st.session_state.seen_ids:
                            st.session_state.seen_ids.add(ident)
                            st.session_state.seen_ids_list.append(ident)
                            st.session_state.fiches_history.append((timestamp, ident))

                            with col2:
                                with st.expander(f"▶️ Fiche de {ident}", expanded=True):
                                    if ident in person_info:
                                        info = person_info[ident]

                                        # Affichage de la photo unique
                                        photo_path = info.get("photo", "")
                                        if photo_path and os.path.exists(photo_path):
                                            st.image(Image.open(photo_path), use_container_width=True)
                                        else:
                                            st.write("_Pas de photo._")

                                        st.markdown(f"**Nom :** {info.get('nom','')}")
                                        st.markdown(f"**Prénom :** {info.get('prenom','')}")
                                        st.markdown(f"**Nationalité :** {info.get('nationalite','')}")
                                        st.markdown(f"**Date de naissance :** {info.get('date_naissance','')}")
                                        st.markdown(f"**Sexe :** {info.get('sexe','')}")
                                        st.markdown(f"**Occupation :** {info.get('occupation','')}")
                                        st.markdown(f"**Email :** {info.get('email','')}")
                                        st.markdown(f"**Commentaires :** {info.get('commentaires','')}")

                                        # Bouton téléchargement JSON
                                        fiche_json = json.dumps(info, ensure_ascii=False, indent=2)
                                        st.download_button(
                                            label="💾 Télécharger la fiche JSON",
                                            data=fiche_json,
                                            file_name=f"{ident}.json",
                                            mime="application/json"
                                        )
                                    else:
                                        st.write("❗ Aucune fiche disponible.")
                else:
                    st.session_state.historique.append((timestamp, "Aucun visage détecté"))
                    st.session_state.dernier_identifiant = "Aucun visage détecté"


# ----------------------------------------------------
# 6. COLONNE DE DROITE : AFFICHAGE PERSISTANT & HISTORIQUES
# ----------------------------------------------------
with col2:
    # 6.a. Fiches déjà affichées
    st.subheader("📋 Fiches déjà affichées")
    if st.session_state.seen_ids_list:
        for ident in st.session_state.seen_ids_list:
            with st.expander(f"▶️ Fiche de {ident}", expanded=False):
                if ident in person_info:
                    info = person_info[ident]

                    # Affichage de la photo unique
                    photo_path = info.get("photo", "")
                    if photo_path and os.path.exists(photo_path):
                        st.image(Image.open(photo_path), use_container_width=True)
                    else:
                        st.write("_Pas de photo._")

                    st.markdown(f"**Nom :** {info.get('nom','')}")
                    st.markdown(f"**Prénom :** {info.get('prenom','')}")
                    st.markdown(f"**Nationalité :** {info.get('nationalite','')}")
                    st.markdown(f"**Date de naissance :** {info.get('date_naissance','')}")
                    st.markdown(f"**Sexe :** {info.get('sexe','')}")
                    st.markdown(f"**Occupation :** {info.get('occupation','')}")
                    st.markdown(f"**Email :** {info.get('email','')}")
                    st.markdown(f"**Commentaires :** {info.get('commentaires','')}")

                    # Bouton téléchargement JSON
                    fiche_json = json.dumps(info, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 Télécharger la fiche JSON",
                        data=fiche_json,
                        file_name=f"{ident}.json",
                        mime="application/json"
                    )
                else:
                    st.write("❗ Aucune fiche trouvée.")
    else:
        st.info("Aucune fiche n’a encore été affichée.")

    st.markdown("---")

    # 6.b. Historique des détections
    st.subheader("🗒️ Historique des détections")
    if st.session_state.historique:
        hist_df = DataFrame(st.session_state.historique[-200:], columns=["Date/Heure", "Identifiant"])
        st.dataframe(hist_df, height=240)
    else:
        st.info("Aucune détection enregistrée.")

    st.markdown("---")

    # 6.c. Historique des fiches installées + boutons « Réinstaller »
    st.subheader("📚 Historique des fiches installées")
    if st.session_state.fiches_history:
        fiches_df = DataFrame(st.session_state.fiches_history, columns=["Date/Heure", "Identifiant de la fiche"])
        st.dataframe(fiches_df, height=200)

        st.write("💡 Réinstaller une fiche depuis l’historique :")
        for idx, (ts, identf) in enumerate(reversed(st.session_state.fiches_history)):
            bouton_key = f"reinstall_{idx}"
            if st.button(f"⟳ Réinstaller « {identf} »", key=bouton_key):
                if identf in person_info:
                    info = person_info[identf]
                    if identf not in st.session_state.seen_ids:
                        st.session_state.seen_ids.add(identf)
                        st.session_state.seen_ids_list.append(identf)
                    with st.expander(f"▶️ (Réinstallée) Fiche de {identf}", expanded=True):
                        photo_path = info.get("photo", "")
                        if photo_path and os.path.exists(photo_path):
                            st.image(Image.open(photo_path), use_container_width=True)
                        else:
                            st.write("_Pas de photo._")

                        st.markdown(f"**Nom :** {info.get('nom','')}")
                        st.markdown(f"**Prénom :** {info.get('prenom','')}")
                        st.markdown(f"**Nationalité :** {info.get('nationalite','')}")
                        st.markdown(f"**Date de naissance :** {info.get('date_naissance','')}")
                        st.markdown(f"**Sexe :** {info.get('sexe','')}")
                        st.markdown(f"**Occupation :** {info.get('occupation','')}")
                        st.markdown(f"**Email :** {info.get('email','')}")
                        st.markdown(f"**Commentaires :** {info.get('commentaires','')}")

                        # Bouton téléchargement JSON
                        fiche_json = json.dumps(info, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="💾 Télécharger la fiche JSON",
                            data=fiche_json,
                            file_name=f"{identf}.json",
                            mime="application/json"
                        )
                else:
                    st.warning(f"Aucune fiche disponible pour « {identf} » !")
    else:
        st.info("Aucune fiche n’a encore été installée.")

