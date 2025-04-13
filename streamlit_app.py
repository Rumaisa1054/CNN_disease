import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# Streamlit app
st.title("TensorFlow Model via Flask API")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the image in the app
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert image to numpy array (for TensorFlow model)
    # Optionally, resize or preprocess the image depending on your model
    image_array = np.array(image)

    # Prepare the payload (convert image array to list for JSON format)
    payload = {'image': image_array.tolist()}

    # Send request to Flask API
    response = requests.post('http://127.0.0.1:5000/predict', json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
    else:
        st.write("Error: Unable to fetch prediction.")
