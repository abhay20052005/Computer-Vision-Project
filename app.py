import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model globably so it's ready for fast predictions
MODEL_PATH = 'cat_dog_pretrained_model.h5'
model = None

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Warning: {MODEL_PATH} not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train first.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the file directly from memory
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        
        # Preprocessing the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Predict
        classes = model.predict(x, batch_size=1)
        score = float(classes[0][0])
        
        if score > 0.5:
            result = "Dog"
            confidence = score
        else:
            result = "Cat"
            confidence = 1.0 - score
            
        return jsonify({
            'success': True,
            'result': result,
            'confidence': confidence * 100
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
