import streamlit as st
from PIL import Image
import torch
from ultralytics import RTDETR 
import cv2
import numpy as np

@st.cache_resource
def load_model():
    model = RTDETR('bestmodel-rtdetrl.pt') 
    return model

model = load_model()

st.title("🦺 PPE Compliance Detection App")
st.markdown("Upload an image to detect **Helmet**, **Vest**, **Gloves**, and **Boots**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection..."):
        results = model(img_bgr)

    annotated_frame = results[0].plot() 

    st.image(annotated_frame, caption="Detection Results", use_column_width=True)

    with st.expander("Detection Details"):
        st.json(results[0].tojson())
