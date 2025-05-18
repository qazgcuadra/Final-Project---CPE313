import streamlit as st
from PIL import Image
import torch
from ultralytics import RTDETR  # Replace with YOLO if you're using that
import cv2
import numpy as np

# Load your trained model (only once, cached)
@st.cache_resource
def load_model():
    model = RTDETR('bestmodel-rtdetrl.pt')  # Replace path if needed
    return model

model = load_model()

# App title
st.title("🦺 PPE Detection App")
st.markdown("Upload an image to detect **Helmet**, **Vest**, **Gloves**, and **Boots** using your custom-trained RT-DETR model.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to numpy array and BGR (OpenCV format)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run inference with confidence threshold set to 0.5
    with st.spinner("Detecting PPE..."):
        results = model(img_bgr, conf=0.5)  # <-- Set confidence here

    # Plot detections
    annotated_frame = results[0].plot()

    # Show annotated result
    st.image(annotated_frame, caption="Detection Results", use_column_width=True)

    # Optional: Show raw detection JSON
    with st.expander("Detection JSON Output"):
        st.json(results[0].tojson())