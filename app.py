import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("🩺 Skin Disease Detection AI")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.h5")

model = load_model()

class_names = ["healthy", "acne", "eczema"]

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = class_names[np.argmax(pred)]

    st.success(f"Prediction: {result}")