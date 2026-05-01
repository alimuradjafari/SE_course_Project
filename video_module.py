import streamlit as st
import tempfile
import cv2
import threading
import os

def show_video_detection(client):
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.header(" Drone Detection in Video")

        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")

        if uploaded_video:
            FRAME_SKIP = 5  # run detection every Nth frame
            latest_detections = []
            lock = threading.Lock()

            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            frame_count = 0
            detection_thread = None

            def run_detection(frame):
                """Run inference in background"""
                nonlocal latest_detections
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                    cv2.imwrite(temp_img.name, frame)
                    temp_path = temp_img.name
                try:
                    result = client.infer(temp_path, model_id="drones_new-h4w1y/1")
                    with lock:
                        latest_detections = result.get("predictions", [])
                except Exception as e:
                    print(f"Detection error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run detection every FRAME_SKIP frames
                if frame_count % FRAME_SKIP == 0 and (detection_thread is None or not detection_thread.is_alive()):
                    detection_thread = threading.Thread(target=run_detection, args=(frame.copy(),))
                    detection_thread.start()

                # Draw detections from the latest inference
                with lock:
                    for det in latest_detections:
                        x, y = int(det["x"]), int(det["y"])
                        w, h = int(det["width"]), int(det["height"])
                        cls, conf = det["class"], det["confidence"]
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls}: {conf:.2%}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the frame in Streamlit
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            cap.release()
            st.success(" Video processing complete!")
            if os.path.exists(video_path):
                os.unlink(video_path)
    #else:
        #st.info(" Upload a video to start real-time drone detection.")

       