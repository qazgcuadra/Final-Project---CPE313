import streamlit as st
from PIL import Image
import torch
from ultralytics import RTDETR
import cv2
import numpy as np
import tempfile
import os
from decord import VideoReader, cpu

st.set_page_config(layout="wide")

# Load your trained model (only once, cached)
@st.cache_resource
def load_model():
    model = RTDETR('bestmodel-rtdetrl.pt')  # Replace path if needed
    return model

model = load_model()

# App title
st.title("🦺 PPE Detection App")
st.markdown("Upload an image or video to detect **Helmet**, **Vest**, **Gloves**, and **Boots** using your custom-trained RT-DETR model.")

# Frame extraction function
def extract_frames(video_path, num_frames=10, size=(640, 640)):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = [cv2.resize(vr[i].asnumpy(), size) for i in indices]
    return frames

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        # Handle image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Run Detection"):
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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        mode = st.radio("Select action for video:", ["Run Detection", "Get Frames"])

        if mode == "Run Detection":
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            st.info("Processing video... please wait")

            # Temporary video writer for output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            with st.spinner("Detecting PPE in video..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=0.5)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    # Convert and display in Streamlit
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, channels="RGB", use_column_width=True)

                cap.release()
                out.release()

            st.success("Video processing complete.")
            st.video(out_path)

        elif mode == "Get Frames":
            st.subheader("Detected Frames")

            num_frames = 10  # Fixed number of frames
            frames = extract_frames(tfile.name, num_frames=num_frames, size=(640, 640))
            frame_cols = st.columns(4)

            for idx, frame in enumerate(frames):
                with frame_cols[idx % 4]:
                    results = model(frame, conf=0.5)
                    result = results[0]
                    detected_frame = result.plot()
                    frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {idx + 1}", use_column_width=True)

                    detected_classes = result.names
                    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                    if class_ids:
                        st.markdown("**Detected Classes:**")
                        for cid in class_ids:
                            st.markdown(f"- {detected_classes[cid]}")
                    else:
                        st.info("No PPE detected in this frame.")