from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('300.keras')  # Adjust path if needed

# Class names
class_names = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Chickenpox',
    4: 'Cowpox',
    5: 'Dermatofibroma',
    6: 'HFMD',
    7: 'Healthy',
    8: 'Measles',
    9: 'Melanocytic nevi',
    10: 'Melanoma',
    11: 'Monkeypox',
    12: 'Squamous cell carcinoma',
    13: 'Vascular lesions'
}

# Prediction function
def predict_skin_lesion(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        class_name = class_names.get(predicted_class, f"Class {predicted_class}")
        return class_name, confidence
    except Exception as e:
        return str(e), 0.0
        
@app.route('/')
def home():
    return "TensorFlow Flask API is running!"

# Flask route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    filepath = os.path.join('temp.jpg')
    file.save(filepath)

    # Make prediction
    class_name, confidence = predict_skin_lesion(filepath)

    # Clean up temp file
    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        'predicted_class': class_name,
        'confidence': round(confidence * 100, 2)
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

