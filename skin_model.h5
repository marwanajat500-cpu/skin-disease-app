import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

st.title("🩺 Skin Disease Detection AI")

# تحميل الموديل
model = load_model("skin_model.h5")

file = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image)

    # تجهيز الصورة
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    # prediction
    prediction = model.predict(img)

    st.success(f"Résultat: {prediction}")
