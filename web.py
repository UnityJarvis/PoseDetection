import streamlit as st
from PIL import Image
import cv2
import numpy as np
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
# from IPython.display import HTML
import modell

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Function to process the image and get the output
def process_image(image):
    # Process the image using your .ipynb code here
    # For this example, we'll just convert it to grayscale using OpenCV
    sample_img = np.array(image)

    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    img_copy = sample_img.copy()
    height, width, _ = img_copy.shape
    landmarks = []
    # Check if any landmarks are found.
    if results.pose_landmarks:

    # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    # Specify a size of the figure.
        fig = plt.figure(figsize = [10, 10])


    return img_copy

def Blackie_image(image):
    img = np.array(image)
    # img = cv2.imread(r"OIP.jpeg")

        # imageWidth, imageHeight = img.shape[:2]

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    blackie = np.zeros(img.shape) # Blank image

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blackie, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # draw landmarks on blackie

    landmarks = results.pose_landmarks.landmark

    blackie_normalized = blackie.astype(float) / 255.0

    return blackie_normalized

def record():

    import cv2
    import mediapipe as mp
    import numpy as np
    import time

    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # List of pose landmarks you want to store
    points = [...]  # Add your desired pose landmarks here

    # Function to capture image and store pose landmarks
    def capture_image_and_landmarks():
        cap = cv2.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras

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

            last_frame = frame  # Update the last_frame with the latest frame

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

        frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
        st.image(frame_bytes, channels="BGR", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

        return last_frame

    # Call the function to capture images for 10 seconds and get the last frame
    last_frame = capture_image_and_landmarks()

    # Now you have the last_frame, and you can do further processing or saving if needed.
    if last_frame is not None:
        # plt.title("sample_Image")
        # plt.axis('off')                       # Removes the labels
        # plt.imshow(last_frame[:,:,::-1])      # Convert BGR to RGB as plt.show() expects RGB
        # plt.show()    
        return last_frame                        # Display the image
    else:
        print("No frames captured.")


def main():
    st.title("Image and Video Analysis")

    # Add a sidebar with options
    option = st.sidebar.selectbox("Select Option", ("Upload Image", "Record Video"))

    if option == "Upload Image":
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Process Image"):
                # Process the image and get the output
                output_image = process_image(image)

                # Display the processed output
                st.image(output_image, caption="Processed Image", use_column_width=True)
            if st.button("Blackie Image"):
                # Process the image and get the output
                out_image = Blackie_image(image)

                # Display the processed output
                st.image(out_image, caption="Blackie Image", use_column_width=True)
                # final = modell.predict_image(out_image)
                # st.write(final)
                # col1, col2 = st.columns(2)
                # col1.image(output_image, caption="Processed Image", use_column_width=True)
                # col2.image(out_image , caption="Blackie Image", use_column_width=True)
            if st.button("Predict Pose"):
                final = modell.predict_image(image)
                st.title("Pose" f"{final}")
    elif option == "Record Video":
        st.subheader("Record a Video")
        if st.button("Start"):
            frame = record()
            if st.button("Process Image"):
                # Process the image and get the output
                output_image = process_image(frame)

                # Display the processed output
                st.image(output_image, caption="Processed Image", use_column_width=True)
            if st.button("Blackie Image"):
                # Process the image and get the output
                out_image = Blackie_image(frame)

                # Display the processed output
                st.image(out_image, caption="Blackie Image", use_column_width=True)
                # final = modell.predict_image(out_image)
                # st.write(final)
                # col1, col2 = st.columns(2)
                # col1.image(output_image, caption="Processed Image", use_column_width=True)
                # col2.image(out_image , caption="Blackie Image", use_column_width=True)
            if st.button("Predict Pose"):
                final = modell.predict_image(frame)
                st.title("Pose" f"{final}")
            


if __name__ == "__main__":
    main()
