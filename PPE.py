import streamlit as st
import cv2
import tempfile
from ultralytics import RTDETR
import numpy as np
from PIL import Image

# Load your trained model
model = RTDETR("bestmodel-rtdetrl.pt")  # Replace with your actual model path

# Required PPE classes to check for safety
required_ppe_classes = ['helmet', 'vest']  # Extend with 'gloves', 'boots' if needed
class_names = model.model.names

# App title
st.markdown("## 👷🦺 PPE Detection Application")

# PPE filter dropdown (optional UI)
selected_class = st.selectbox("Select PPE class to filter", ["all"] + required_ppe_classes)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video for PPE detection",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mpeg4"]
)

# Function to check for missing PPE items
def check_missing_ppe(results):
    boxes = results[0].boxes.data.cpu().numpy()
    detected_classes = set()

    for box in boxes:
        class_id = int(box[5])
        class_name = class_names[class_id]
        detected_classes.add(class_name)

    missing_items = [item for item in required_ppe_classes if item not in detected_classes]
    return missing_items

# Process uploaded image
def process_image(image):
    image_array = np.array(image)
    results = model(image_array)

    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detection Results", use_column_width=True)

    missing = check_missing_ppe(results)
    if not missing:
        st.success("✅ Scene is SAFE: All required PPE detected.")
    else:
        st.error(f"❌ Scene is UNSAFE: Missing PPE - {', '.join(missing)}")

# Process uploaded video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not read the video file.")
        return

    results = model(frame)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_frame, channels="RGB", use_column_width=True, caption="First Frame Detection")

    missing = check_missing_ppe(results)
    if not missing:
        st.success("✅ SAFE: All required PPE detected in the first frame.")
    else:
        st.success(f"✅ SAFE: All required PPE detected in the first frame.")

    cap.release()

# Main logic
if uploaded_file:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        process_image(image)

    elif file_type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)