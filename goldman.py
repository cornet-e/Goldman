import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Goldmann – Analyse Interactive", layout="wide")

st.title("👁️ Interpréteur de champ visuel Goldmann")
st.write("Cliquez sur le centre et le point 90° directement sur l'image pour calibrer l'échelle. Les isoptères seront détectées automatiquement.")

# --- Upload image ---
uploaded_file = st.file_uploader("Choisir une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lecture image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image_rgb.shape

    st.subheader("Image avec points de calibration")
    
    # --- Canvas interactif ---
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="red",
        background_image=Image.fromarray(image_rgb),
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="point",
        key="canvas"
    )

    if canvas_result.json_data is not None:
        points = canvas_result.json_data["objects"]
        if len(points) >= 2:
            # Récupération des deux premiers points
            center_x, center_y = points[0]["left"], points[0]["top"]
            point_x, point_y = points[1]["left"], points[1]["top"]

            # Affichage des points
            st.write(f"Centre : ({center_x:.1f}, {center_y:.1f})")
            st.write(f"Point 90° : ({point_x:.1f}, {point_y:.1f})")

            # --- Calcul de l'échelle ---
            d_pixels = np.sqrt((point_x - center_x)**2 + (point_y - center_y)**2)
            if d_pixels > 0:
                scale = 90 / d_pixels
                st.success(f"Calibration : 1 pixel = {scale:.3f}°")
            else:
                scale = None
                st.warning("Distance nulle, choisissez deux points différents.")

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

            contours_filtered = [
                c for c in contours
                if cv2.contourArea(c) > 100 and circularity(c) > 0.3
            ]

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
            st.info("Cliquez sur deux points pour calibrer (centre et 90°).")
else:
    st.info("➡️ Téléversez une image pour commencer")
