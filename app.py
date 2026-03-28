import streamlit as st
import numpy as np
from PIL import Image

st.title("🩺 Skin Disease App (Demo)")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    st.success("Demo mode: model not loaded yet")
