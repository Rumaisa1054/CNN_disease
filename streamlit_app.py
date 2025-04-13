import streamlit as st
import requests
import numpy as np

# Streamlit app
st.title("TensorFlow Model via Flask API")

# Input form
input_data = st.text_input("Enter input data (comma separated)")

if input_data:
    # Convert input to numpy array
    input_array = np.array([float(x) for x in input_data.split(',')])

    # Prepare the payload
    payload = {'input': input_array.tolist()}

    # Send request to Flask API
    response = requests.post('http://127.0.0.1:5000/predict', json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
    else:
        st.write("Error: Unable to fetch prediction.")
