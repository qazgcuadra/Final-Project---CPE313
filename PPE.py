import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR
import tempfile
import os
import torch

st.set_page_config(layout="wide")
st.title("PPE Detection from Image or Video")

model = RTDETR("bestmodel-rtdetrl.pt")  # Update with your trained model path

# Fixed PPE classes to filter by
ppe_choices = ["all", "helmet", "vest", "gloves", "boots"]
selected_ppe = st.selectbox("Select PPE class to filter", ppe_choices, index=0)

# Slider (or radio) to select input type
input_type = st.radio("Choose input type:", ["Image", "Video"])

def extract_frames(video_path, num_frames=10, size=(640, 640)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        st.warning("Video has fewer frames than the requested number. Reducing to available frames.")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()
    return frames

def filter_and_draw(frame, result, selected_ppe):
    if selected_ppe != "all":
        detected_class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        names = result.names
        filtered_indices = [idx for idx, cid in enumerate(detected_class_ids) if names[cid].lower() == selected_ppe.lower()]
        if filtered_indices:
            boxes = result.boxes[filtered_indices]
            detected_frame = frame.copy()
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                cls_id = int(box.cls.cpu().numpy())
                conf = box.conf.cpu().numpy()
                label = f"{names[cls_id]} {conf[0]:.2f}"
                cv2.rectangle(detected_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                cv2.putText(detected_frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            detected_frame = frame.copy()
    else:
        detected_frame = result.plot()
    return detected_frame

if input_type == "Video":
    uploaded_file = st.file_uploader("Upload a video for PPE detection", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)
        st.subheader("Detected Frames with PPE Items")

        num_frames = 10
        frames = extract_frames(video_path, num_frames=num_frames, size=(640, 640))

        for i in range(0, len(frames), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(frames):
                    with cols[j]:
                        frame = frames[i + j]
                        results = model(frame, conf=0.5)
                        result = results[0]
                        detected_frame = filter_and_draw(frame, result, selected_ppe)
                        frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {i + j + 1}", use_container_width=True)

                        # Show detected classes
                        if selected_ppe == "all":
                            class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                            if len(class_ids) > 0:
                                detected_classes = result.names
                                st.markdown("**Detected Classes:**")
                                for cid in class_ids:
                                    st.markdown(f"- {detected_classes[cid]}")
                            else:
                                st.info("No PPE detected in this frame.")
                        else:
                            detected_class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                            names = result.names
                            filtered_indices = [idx for idx, cid in enumerate(detected_class_ids) if names[cid].lower() == selected_ppe.lower()]
                            if filtered_indices:
                                st.markdown(f"**Detected Classes:** - {selected_ppe}")
                            else:
                                st.info(f"No {selected_ppe} detected in this frame.")

elif input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image for PPE detection", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model(frame, conf=0.5)
        result = results[0]
        detected_frame = filter_and_draw(frame, result, selected_ppe)
        frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detected PPE", use_container_width=True)

        # Show detected classes
        if selected_ppe == "all":
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
            if len(class_ids) > 0:
                detected_classes = result.names
                st.markdown("**Detected Classes:**")
                for cid in class_ids:
                    st.markdown(f"- {detected_classes[cid]}")
            else:
                st.info("No PPE detected in this image.")
        else:
            detected_class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
            names = result.names
            filtered_indices = [idx for idx, cid in enumerate(detected_class_ids) if names[cid].lower() == selected_ppe.lower()]
            if filtered_indices:
                st.markdown(f"**Detected Classes:** - {selected_ppe}")
            else:
                st.info(f"No {selected_ppe} detected in this image.")