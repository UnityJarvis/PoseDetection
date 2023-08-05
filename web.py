import streamlit as st
from PIL import Image
import cv2
import numpy as np
import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import modell

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def process_image(image):
    print("Func")
    sample_img = np.array(image)

    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    img_copy = sample_img.copy()
    height, width, _ = img_copy.shape
    landmarks = []
    if results.pose_landmarks:

        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:

                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        fig = plt.figure(figsize = [10, 10])


    return img_copy

def Blackie_image(image):
    img = np.array(image)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blackie = np.zeros(img.shape)

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blackie, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmarks = results.pose_landmarks.landmark

    blackie_normalized = blackie.astype(float) / 255.0

    return blackie_normalized

def record():

    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    import time

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    points = mp_pose.PoseLandmark       

    data = []

    for p in points:
            x = str(p)[13:]
            data.append(x + "_x")
            data.append(x + "_y")
            data.append(x + "_z")
            data.append(x + "_vis")
    data = pd.DataFrame(columns = data) 

    def capture_image_and_landmarks():
        cap = cv2.VideoCapture(0)  

        start_time = time.time()
        last_frame = None

        while time.time() - start_time <= 10:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            last_frame = frame  

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

        frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
        st.image(frame_bytes, channels="BGR", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

        return last_frame

    last_frame = capture_image_and_landmarks()

    if last_frame is not None:   
        return last_frame                  
    else:
        print("No frames captured.")


def main():
    st.title("Image and Video Analysis")

    option = st.sidebar.selectbox("Select Option", ("Upload Image", "Record Video"))

    if option == "Upload Image":
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:

            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Process Image"):
                output_image = process_image(image)
                st.image(output_image, caption="Processed Image", use_column_width=True)
            if st.button("Blackie Image"):
                out_image = Blackie_image(image)

                st.image(out_image, caption="Blackie Image", use_column_width=True)

            if st.button("Predict Pose"):
                final = modell.predict_image(image)
                st.title("Pose" f"{final}")
    elif option == "Record Video":
        st.subheader("Record a Video")
        if st.button("Start"):
            frame = record()
            output_image = process_image(frame)

            st.image(output_image, caption="Processed Image", use_column_width=True)

            out_image = Blackie_image(frame)

            st.image(out_image, caption="Blackie Image", use_column_width=True)

            final = modell.predict_image(frame)
            st.title("Pose" f"{final}")


if __name__ == "__main__":
    main()
