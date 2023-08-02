import streamlit as st
from data_processing import process_data_and_extract_features

def main():
    st.title("Pose Detection and Analysis Website")

    st.sidebar.header("Upload Image")
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        if st.sidebar.button("Analyze"):
            result = process_data_and_extract_features(uploaded_image)
            st.write("Processed Result:")
            st.write(result)
