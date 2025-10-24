import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interpréteur de champ visuel de Goldmann", layout="wide")

st.title("👁️ Interpréteur automatique de champ visuel de Goldmann")
st.write("Téléversez une image scannée ou une photo d’un champ visuel de Goldmann pour une analyse automatique.")

# === Upload de l'image ===
uploaded_file = st.file_uploader("Choisir une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lecture de l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Image originale")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # === Traitement de l'image ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

    st.subheader("Contours détectés")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    # === Analyse automatique ===
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radii = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            r = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            radii.append(r)

    st.subheader("Résultats d’analyse")

    if radii:
        mean_radius = np.mean(radii)
        std_radius = np.std(radii)
        st.write(f"**Rayon moyen des isoptères :** {mean_radius:.1f} pixels")
        st.write(f"**Variabilité des rayons :** {std_radius:.1f} pixels")

        # Interprétation simple
        interpretation = ""
        if mean_radius < 0.4 * max(image.shape):
            interpretation = "⚠️ Rétrécissement concentrique du champ visuel suspecté."
        elif std_radius > 0.3 * mean_radius:
            interpretation = "⚠️ Asymétrie des isoptères — possible scotome ou hémianopsie."
        else:
            interpretation = "✅ Champ visuel globalement normal."

        st.markdown(f"### 🧠 Interprétation : {interpretation}")
    else:
        st.warning("Aucune isoptère détectée — essayez une image plus contrastée ou mieux centrée.")
else:
    st.info("➡️ Téléversez une image pour commencer l’analyse.")

# === Pied de page ===
st.markdown("---")
st.caption("Prototype Streamlit – Analyse automatique du champ visuel de Goldmann")