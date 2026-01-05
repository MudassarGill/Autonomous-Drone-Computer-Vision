import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

st.set_page_config(page_title="Drone Detection", layout="centered")
st.title("🚁 Drone Real-Time Object Detection")

st.write("✅ App started")

@st.cache_resource
def load_model():
    st.write("✅ Loading YOLO model...")
    return YOLO("yolov8n.pt")

model = load_model()
st.write("✅ YOLO loaded")

video_file = st.file_uploader("📂 Upload video", type=["mp4", "avi", "mov"])

FRAME_W, FRAME_H = 640, 360

video_box = st.empty()
decision_box = st.empty()

if video_file is None:
    st.warning("⚠ Please upload a video file")
else:
    st.success("✅ Video uploaded")

    with open("drone_video_small.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("drone_video_small.mp4")

    if not cap.isOpened():
        st.error("❌ Video not opened")
    else:
        st.success("🎬 Video playing")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("✅ Video finished")
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            decision = "GO FORWARD"
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].xyxy[0]
                cx = (x1 + x2) / 2

                if cx < FRAME_W * 0.4:
                    decision = "MOVE RIGHT"
                elif cx > FRAME_W * 0.6:
                    decision = "MOVE LEFT"
                else:
                    decision = "STOP"

            decision_box.markdown(f"### 🧠 Decision: **{decision}**")

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_box.image(annotated)

            time.sleep(0.03)

        cap.release()
