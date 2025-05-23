import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR
import tempfile
import torch

st.set_page_config(layout="wide")
st.title("👷🦺 PPE Detection Application")

model = RTDETR("bestmodel-rtdetrl.pt")  # Update with your trained model path

ppe_choices = ["all", "helmet", "vest"]
selected_ppe = st.selectbox("Select PPE class to filter", ppe_choices, index=0)

# Extract a single frame (middle) from the video
def extract_single_frame(video_path, size=(640, 640)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        st.error("Unable to extract frames from the video.")
        return None

    middle_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to read the selected frame.")
        return None

    return cv2.resize(frame, size)

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1, yi1 = max(x1, a1), max(y1, b1)
    xi2, yi2 = min(x2, a2), min(y2, b2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

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

uploaded_file = st.file_uploader(
    "Upload an image or video for PPE detection",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext in ["mp4", "avi", "mov"]:
        # Handle video
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)

        frame = extract_single_frame(video_path, size=(640, 640))
        if frame is not None:
            results = model(frame, conf=0.5)
            result = results[0]
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            names = result.names
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            detected_objects = []
            for i, cls_id in enumerate(class_ids):
                label = names[cls_id].lower()
                coords = xyxy[i]
                detected_objects.append({'label': label, 'coords': coords})

            persons = [obj for obj in detected_objects if obj['label'] == 'person']
            helmets = [obj for obj in detected_objects if obj['label'] == 'helmet']
            vests = [obj for obj in detected_objects if obj['label'] == 'vest']

            safe = True
            for person in persons:
                has_helmet = any(iou(person['coords'], helmet['coords']) > 0.3 for helmet in helmets)
                has_vest = any(iou(person['coords'], vest['coords']) > 0.3 for vest in vests)
                if not (has_helmet and has_vest):
                    safe = False
                    break

            if safe and persons:
                st.success("✅ Safe: All detected persons are wearing both helmet and vest.")
            else:
                st.error("❌ Unsafe: One or more detected persons are missing helmet or vest.")

            detected_frame = filter_and_draw(frame, result, selected_ppe)
            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Analyzed Frame", use_container_width=True)

    elif file_ext in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.5)
        result = results[0]
        detected_frame = filter_and_draw(frame, result, selected_ppe)
        frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detected PPE", use_container_width=True)

        # Show detected classes in this filtered view
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
            if len(result.boxes) > 0:
                detected_class_ids = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names
                filtered_indices = [idx for idx, cid in enumerate(detected_class_ids) if names[cid].lower() == selected_ppe.lower()]
                if filtered_indices:
                    st.markdown(f"**Detected Classes:** - {selected_ppe}")
                else:
                    st.info(f"No {selected_ppe} detected in this image.")
            else:
                st.info(f"No {selected_ppe} detected in this image.")

    else:
        st.error("Unsupported file format. Please upload an image or video file.")