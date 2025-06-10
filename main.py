import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# Load YOLOv8 model
model = YOLO("./runs/weights/best.pt")

# Streamlit App Configuration
st.set_page_config(
    page_title="Railway Track Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Settings")
folder_path = st.sidebar.text_input("📂 Enter Folder Path:")

# App Header
st.title("🚂 Railway Track Defect Detection")
st.write(
    "Analyze railway track images for potential defects using the YOLOv8 model. "
    "This app organizes results in an easy-to-read format."
)

# Session state initialization
if "summary_data" not in st.session_state:
    st.session_state.summary_data = []
if "image_files" not in st.session_state:
    st.session_state.image_files = []

# Image Processing
if folder_path:
    if os.path.isdir(folder_path):
        st.success(f"📁 Found folder: {folder_path}")
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]

        if image_files:
            # Only run inference if not already processed
            if not st.session_state.image_files or st.session_state.image_files != image_files:
                st.info(f"🔍 Processing {len(image_files)} image(s)...")
                st.session_state.summary_data = []
                st.session_state.image_files = image_files

                tabs = st.tabs(image_files)

                for idx, image_file in enumerate(image_files):
                    image_path = os.path.join(folder_path, image_file)
                    image = Image.open(image_path)

                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    results = model(image_bgr)
                    annotated_image = results[0].plot()

                    detections = []
                    for detection in results[0].boxes:
                        label = int(detection.cls)
                        confidence = float(detection.conf)
                        detections.append((label, confidence))

                    detection_status = "Defects Detected" if detections else "No Defects"

                    st.session_state.summary_data.append({
                        "Image Name": image_file,
                        "Status": detection_status,
                        "Detections": ", ".join(
                            [f"Label: {d[0]}, Confidence: {d[1]:.2f}" for d in detections]
                        )
                    })

                    with tabs[idx]:
                        st.subheader(f"Image: {image_file}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_container_width=True)
                        with col2:
                            st.image(annotated_image, caption="Detected Image", use_container_width=True)
                        if detections:
                            st.markdown("### Detection Details")
                            for label, confidence in detections:
                                st.write(f"- **Label**: {label}, **Confidence**: {confidence:.2f}")
            else:
                st.success("✅ Already processed images. Showing results from cache.")

            # Show Detection Summary
            st.markdown("## 📝 Detection Summary")
            summary_df = pd.DataFrame(st.session_state.summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Accuracy Calculator
            st.markdown("## 🎯 Accuracy Calculator")
            total_images = st.number_input("Enter total number of images:", min_value=1, value=len(st.session_state.image_files))
            correct_predictions = st.number_input("Enter number of correct predictions:", min_value=0, value=0)

            if st.button("Calculate Accuracy"):
                accuracy = (correct_predictions / total_images) * 100
                st.success(f"📊 Accuracy: **{accuracy:.2f}%**")

        else:
            st.warning("⚠️ No valid images found in the folder.")
    else:
        st.error("🚫 The specified folder does not exist.")
else:
    st.info("📝 Please enter a folder path to get started.")