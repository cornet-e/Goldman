import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Interpréteur de champ visuel de Goldmann", layout="wide")

st.title("👁️ Interpréteur automatique de champ visuel de Goldmann")
st.write(
    """
    Téléversez une image scannée ou une photo d’un champ visuel de **Goldmann**.
    L’application détecte automatiquement les **isoptères colorées**, ignore les axes noirs,
    et fournit une **interprétation automatique** du champ visuel.
    """
)

# === Upload de l'image ===
uploaded_file = st.file_uploader("Choisir une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lecture de l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Image originale")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- Étape 1 : Filtrage couleur (isoptères non noires) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masque général : saturation élevée = couleur
    mask_color = cv2.inRange(hsv, (0, 60, 50), (180, 255, 255))

    # Optionnel : lisser un peu le masque
    mask_color = cv2.medianBlur(mask_color, 5)

    st.subheader("Masque des zones colorées (isoptères potentielles)")
    st.image(mask_color, use_column_width=True, clamp=True)

    # --- Étape 2 : Détection des contours sur le masque ---
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Étape 3 : Filtrage géométrique (enlever les bruits et les lignes droites) ---
    def circularity(contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return 4 * np.pi * area / (perimeter ** 2)

    contours_filtered = [
        c for c in contours
        if cv2.contourArea(c) > 100 and circularity(c) > 0.3
    ]

    # --- Étape 4 : Visualisation des isoptères détectées ---
    output = image.copy()
    cv2.drawContours(output, contours_filtered, -1, (0, 0, 255), 2)

    st.subheader("Isoptères détectées (zones colorées et circulaires)")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- Étape 5 : Analyse automatique ---
    st.subheader("Résultats d’analyse")
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radii = []

    for cnt in contours_filtered:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            r = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            radii.append(r)

    if radii:
        mean_radius = np.mean(radii)
        std_radius = np.std(radii)
        st.write(f"**Rayon moyen des isoptères :** {mean_radius:.1f} pixels")
        st.write(f"**Variabilité des rayons :** {std_radius:.1f} pixels")

        # --- Interprétation simple ---
        if mean_radius < 0.4 * max(image.shape):
            interpretation = "⚠️ Rétrécissement concentrique du champ visuel suspecté."
        elif std_radius > 0.3 * mean_radius:
            interpretation = "⚠️ Asymétrie des isoptères — possible scotome ou hémianopsie."
        else:
            interpretation = "✅ Champ visuel globalement normal."

        st.markdown(f"### 🧠 Interprétation : {interpretation}")
    else:
        st.warning("Aucune isoptère détectée — essayez une image plus contrastée ou plus colorée.")
else:
    st.info("➡️ Téléversez une image pour commencer l’analyse.")

# --- Pied de page ---
st.markdown("---")
st.caption("Prototype Streamlit – Analyse automatique des champs visuels de Goldmann (v1.2)")