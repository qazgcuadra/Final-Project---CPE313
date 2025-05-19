import streamlit as st
from PIL import Image
import torch
from ultralytics import RTDETR
import cv2
import numpy as np
import tempfile
import os

# Load your trained model (only once, cached)
@st.cache_resource
def load_model():
    model = RTDETR('bestmodel-rtdetrl.pt')  # Replace path if needed
    return model

model = load_model()

# App title
st.title("🦺 PPE Detection App")
st.markdown("Upload an image or video to detect common Personal Protective Equipment.")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        # Handle image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert to OpenCV BGR format
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run detection
        with st.spinner("Detecting PPE in image..."):
            results = model(img_bgr, conf=0.5)

        # Annotate and show
        annotated_frame = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Results", use_column_width=True)

        with st.expander("Detection JSON Output"):
            st.json(results[0].tojson())

    elif file_type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        st.info("Processing first 10 frames of video...")

        # Store annotated frames and results
        frame_count = 0
        max_frames = 10
        annotated_frames = []
        detection_jsons = []

        with st.spinner("Detecting PPE in video..."):
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.5)
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_frames.append(annotated_rgb)
                detection_jsons.append(results[0].tojson())

                frame_count += 1

            cap.release()

        st.success("Processed first 10 frames.")

        # Display each frame individually
        st.subheader("🔍 Frame-by-Frame Analysis (First 10 Frames)")
        for i, (img, det_json) in enumerate(zip(annotated_frames, detection_jsons)):
            st.markdown(f"### Frame {i+1}")
            st.image(img, caption=f"Annotated Frame {i+1}", use_column_width=True)
            with st.expander(f"Detection JSON Output for Frame {i+1}"):
                st.json(det_json)