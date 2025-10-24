import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(page_title="Goldmann – Analyse et Calibration", layout="wide")

st.title("👁️ Interpréteur calibré de champ visuel de Goldmann")
st.write("""
Téléversez une image de champ visuel de **Goldmann**, cliquez pour calibrer,
et laissez l’application détecter les **isoptères colorées** et fournir une interprétation.
""")

# === Upload de l'image ===
uploaded_file = st.file_uploader("Choisir une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Étape 1 : Calibration de l’échelle en degrés")

    st.write("""
    👉 Cliquez sur **le centre de fixation** puis sur **le cercle à 90°** sur l’image ci-dessous.
    Cela permet de convertir les pixels en degrés visuels.
    """)

    # Convertir np.array RGB en PIL
    image_pil = Image.fromarray(image_rgb)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="red",
        background_image=image_pil,  # <-- PIL.Image ici
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        points = canvas_result.json_data["objects"]
        if len(points) == 2:
            x1, y1 = points[0]["left"], points[0]["top"]
            x2, y2 = points[1]["left"], points[1]["top"]
            d_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            scale = 90 / d_pixels
            st.success(f"Calibration réussie : 1 pixel = {scale:.3f}°")
        else:
            scale = None
            st.info("Veuillez cliquer sur **2 points** pour calibrer.")
    else:
        scale = None

    # --- Étape 2 : Détection couleur (isoptères) ---
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

    contours_filtered = [
        c for c in contours
        if cv2.contourArea(c) > 100 and circularity(c) > 0.3
    ]

    output = image.copy()
    cv2.drawContours(output, contours_filtered, -1, (0, 0, 255), 2)

    st.subheader("Isoptères détectées")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- Étape 3 : Analyse en degrés ---
    if scale and len(contours_filtered) > 0:
        center = (int(x1), int(y1))  # centre calibré choisi par l’utilisateur
        radii_deg = []

        for cnt in contours_filtered:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r_pix = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                r_deg = r_pix * scale
                radii_deg.append(r_deg)

        mean_r = np.mean(radii_deg)
        std_r = np.std(radii_deg)

        st.subheader("Résultats calibrés")
        st.write(f"**Rayon moyen des isoptères :** {mean_r:.1f}°")
        st.write(f"**Variabilité :** {std_r:.1f}°")

        # --- Interprétation automatique ---
        if mean_r < 40:
            interpretation = "⚠️ Champ visuel rétréci (rétrecissement concentrique probable)."
        elif std_r > 0.3 * mean_r:
            interpretation = "⚠️ Asymétrie importante — scotome ou hémianopsie possible."
        else:
            interpretation = "✅ Champ visuel globalement normal."

        st.markdown(f"### 🧠 Interprétation : {interpretation}")
    elif scale:
        st.warning("Aucune isoptère détectée.")
    else:
        st.info("Veuillez d’abord calibrer l’image (2 clics sur la figure).")
else:
    st.info("➡️ Téléversez une image pour commencer.")

st.markdown("---")
st.caption("Prototype Streamlit – Interprétation calibrée du champ visuel de Goldmann (v2.0)")