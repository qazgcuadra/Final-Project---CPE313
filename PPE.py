import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR
import tempfile
import os
import torch

st.set_page_config(layout="wide")
st.title("PPE Detection from Video")

model = RTDETR("bestmodel-rtdetrl.pt")  # Update with your trained model path

# Function to extract fixed number of resized frames from video
def extract_frames(video_path, num_frames=10, size=(640, 640)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)
    
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)
        if len(frames) >= num_frames:
            break

    cap.release()
    return frames

uploaded_file = st.file_uploader("Upload a video for PPE detection", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)
    st.subheader("Detected Frames with PPE Items")

    # Process frames and run inference
    num_frames = 10
    frames = extract_frames(video_path, num_frames=num_frames, size=(640, 640))

    for i in range(0, len(frames), 4):  # Display 4 frames per row
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(frames):
                with cols[j]:
                    frame = frames[i + j]
                    results = model(frame, conf=0.5)
                    result = results[0]
                    detected_frame = result.plot()
                    frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {i + j + 1}", use_container_width=True)

                    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                    if class_ids:
                        detected_classes = result.names
                        st.markdown("**Detected Classes:**")
                        for cid in class_ids:
                            st.markdown(f"- {detected_classes[cid]}")
                    else:
                        st.info("No PPE detected in this frame.")