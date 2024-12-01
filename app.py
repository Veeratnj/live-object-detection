import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
@st.cache_resource  # Cache model to avoid reloading
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Streamlit UI
st.title("Real-time Object Detection using YOLOv5")
st.sidebar.write("Adjust model settings:")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Upload image/video or use webcam
source_option = st.sidebar.selectbox("Select Source", ["Upload Image", "Upload Video", "Webcam"])

if source_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Detect objects
        results = model(image)
        st.image(np.squeeze(results.render()), caption='Detection Results', use_column_width=True)
        st.write(results.pandas().xyxy[0])  # Display detection results in a table

elif source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Video processing isn't implemented in this example.")

elif source_option == "Webcam":
    st.write("Webcam processing requires local Streamlit execution.")
    run = st.checkbox("Start Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            result_frame = np.squeeze(results.render())
            stframe.image(result_frame, channels='RGB')
        cap.release()
        cv2.destroyAllWindows()
