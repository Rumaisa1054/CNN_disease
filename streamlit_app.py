import subprocess
import time
import streamlit as st
import requests
import numpy as np
from PIL import Image

# Function to start Flask API in the background
def start_flask_api():
    subprocess.Popen(["python", "api.py"])

# Check if Flask API is already running, if not, start it
try:
    response = requests.get("http://127.0.0.1:5000/")
    if response.status_code == 200:
        st.write("Flask API is already running.")
except requests.exceptions.RequestException:
    st.write("Flask API not running. Starting it now...")
    start_flask_api()
    time.sleep(5)  # Increased sleep time to ensure Flask has enough time to start

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array (for TensorFlow model)
    image_array = np.array(image)

    # Prepare the payload to send as JSON (convert image to list)
    payload = {'image': image_array.tolist()}

    # Send POST request to Flask API
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
        else:
            st.write("Error: Unable to fetch prediction.")
    except requests.exceptions.RequestException as e:
        st.write(f"Error: {e}")
