import streamlit as st
import tempfile
import cv2
from PIL import Image
import os

def show_image_detection(client):
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.header(" Drone Detection in Images")

        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            img = cv2.imread(temp_path)
            result = client.infer(temp_path, model_id="drones_new-h4w1y/1")

            # Draw bounding boxes
            for detection in result["predictions"]:
                x = int(detection["x"])
                y = int(detection["y"])
                w = int(detection["width"])
                h = int(detection["height"])
                conf = detection["confidence"]
                cls = detection["class"]

                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{cls}: {conf:.2%}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Detected Drones ({len(result['predictions'])} found)", use_container_width=True)
            st.success(f"Detection complete! {len(result['predictions'])} drone(s) found.")
    #else:
        #st.info(" Upload an image to test drone detection.")

