import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("🩺 Skin Disease Detection AI")

# تحميل الموديل
model = load_model("skin_model.h5")

# لائحة الكلاسات (بدليها حسب data ديالك)
classes = ["Acne", "Eczema", "Melanoma", "Normal"]

# upload image
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image choisie", use_column_width=True)

    # preprocessing
    img = image.resize((224, 224))  # حسب الموديل ديالك
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"🧠 Result: {predicted_class}")
