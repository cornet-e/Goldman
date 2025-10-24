import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Goldmann – Analyse Interactive", layout="wide")

st.title("👁️ Interpréteur de champ visuel Goldmann")
st.write("""
Téléversez une image de champ visuel.  
Choisissez le centre et un point sur le cercle 90° via sliders pour calibrer l'échelle, puis cliquez sur **Analyser**.
""")

# --- Upload image ---
uploaded_file = st.file_uploader("Choisir une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lecture image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    st.subheader("Image originale")
    st.image(image_rgb, use_column_width=True)

    st.subheader("Calibration de l'échelle (degrés)")

    col1, col2 = st.columns(2)
    with col1:
        center_x = st.slider("Centre X", 0, w-1, w//2)
        center_y = st.slider("Centre Y", 0, h-1, h//2)
    with col2:
        point_x = st.slider("Point 90° X", 0, w-1, w//2)
        point_y = st.slider("Point 90° Y", 0, h-1, h//2)

    # --- Affichage des points sur l'image ---
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.plot(center_x, center_y, 'ro', label="Centre")
    ax.plot(point_x, point_y, 'go', label="Point 90°")
    ax.legend()
    ax.axis('off')
    st.pyplot(fig)

    # --- Bouton pour lancer l'analyse ---
    if st.button("Analyser"):
        # Calcul de l'échelle
        d_pixels = np.sqrt((point_x - center_x)**2 + (point_y - center_y)**2)
        if d_pixels > 0:
            scale = 90 / d_pixels
            st.success(f"Calibration : 1 pixel = {scale:.3f}°")
        else:
            scale = None
            st.warning("Distance nulle, choisissez des points différents.")

        # --- Détection couleur (isoptères) ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, (0, 60, 50), (180, 255, 255))
        mask_color = cv2.medianBlur(mask_color, 5)

        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def circularity(c):
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                return 0
            return 4 * np.pi * area / (perimeter ** 2)

        contours_filtered = [c for c in contours if cv2.contourArea(c) > 100 and circularity(c) > 0.3]

        # --- Affichage des isoptères ---
        output = image.copy()
        cv2.drawContours(output, contours_filtered, -1, (255, 0, 0), 2)
        st.subheader("Isoptères détectées")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

        # --- Analyse en degrés ---
        if scale and len(contours_filtered) > 0:
            radii_deg = []
            center = (center_x, center_y)
            for cnt in contours_filtered:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    r_pix = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                    radii_deg.append(r_pix * scale)

            mean_r = np.mean(radii_deg)
            std_r = np.std(radii_deg)

            st.subheader("Résultats calibrés")
            st.write(f"Rayon moyen : {mean_r:.1f}°")
            st.write(f"Variabilité : {std_r:.1f}°")

            if mean_r < 40:
                interp = "⚠️ Rétrécissement concentrique probable"
            elif std_r > 0.3 * mean_r:
                interp = "⚠️ Asymétrie importante — possible scotome ou hémianopsie"
            else:
                interp = "✅ Champ visuel globalement normal"
            st.markdown(f"### 🧠 Interprétation : {interp}")
        elif scale:
            st.warning("Aucune isoptère détectée")
        else:
            st.info("Veuillez calibrer correctement les points")
else:
    st.info("➡️ Téléversez une image pour commencer")
