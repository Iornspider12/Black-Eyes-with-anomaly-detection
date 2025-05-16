import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load YOLO model
MODEL_PATH = r"D:\anomaly\runs\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)
model.model.names = {0: 'smoke', 1: 'fire', 2: 'knife'}

st.title("🔥 Anomaly Detection")

# Initialize session state for webcam
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "cap" not in st.session_state:
    st.session_state.cap = None

input_mode = st.selectbox("Choose input source:", ["Webcam", "Sample Image"], key="mode_selector")

# -------------------------- WEBCAM DETECTION --------------------------
if input_mode == "Webcam":
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Webcam Detection"):
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.run_webcam = True

    if col2.button("⏹️ Stop Webcam Detection"):
        st.session_state.run_webcam = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        cv2.destroyAllWindows()

    stframe = st.empty()

    # Run detection if active
    if st.session_state.run_webcam and st.session_state.cap:
        while st.session_state.run_webcam:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("❌ Failed to access webcam.")
                break

            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.model.names.get(cls, "Unknown")
                    label = f"{class_name}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

# -------------------------- STATIC IMAGE DETECTION --------------------------
else:
    image_choice = st.slider("Select Image:", 1, 3, 1, format="Image %d")
    image_path = f"sample_images/{['fire', 'smoke', 'knife'][image_choice - 1]}.jpg"

    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = model.model.names.get(cls, "Unknown")
                label = f"{class_name}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")
    else:
        st.error("Image not found!")
