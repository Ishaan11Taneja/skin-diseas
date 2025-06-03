import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import cv2
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model and class names
model = joblib.load('skin_disease_model.joblib')
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img.flatten()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            img = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict([img])[0]
            probability = model.predict_proba([img])[0]
            
            # Get the predicted class and confidence
            predicted_class = class_names[prediction]
            confidence = float(probability[prediction])
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(class_names, probability)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 