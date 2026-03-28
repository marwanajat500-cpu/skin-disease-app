import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_disease_model.h5")

# أسماء الأمراض (بدليهم إذا مختلفين عندك)
class_names = ["healthy", "acne", "eczema"]

st.title("🩺 Skin Disease Detection AI")

uploaded_file = st.file_uploader("Upload image")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = class_names[np.argmax(pred)]

    st.success("Prediction: " + result)