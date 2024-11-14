# app.py
from flask import Flask, render_template, Response, jsonify
from keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Define file paths
MODEL_PATH = r"C:\Users\rasul\Documents\Python\End\keras_model.h5"
LABELS_PATH = r"C:\Users\rasul\Documents\Python\End\labels.txt"

# Global variables
model = None
class_names = None
camera = None

def load_keras_model():
    """Load the Keras model and labels."""
    global model, class_names
    
    custom_objects = {
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
    }
    
    model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    class_names = open(LABELS_PATH, "r").readlines()

def preprocess_image(image):
    """Preprocess the image for model input."""
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

def generate_frames():
    """Generate frames from webcam with predictions."""
    global camera
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame for display
        processed_image = preprocess_image(frame)
        prediction = model.predict(processed_image, verbose=0)
        
        # Get prediction results
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Add text to frame
        text = f"{class_name[2:].strip()} ({confidence_score*100:.0f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Get latest prediction."""
    if camera is None:
        return jsonify({'error': 'Camera not initialized'})
    
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    processed_image = preprocess_image(frame)
    prediction = model.predict(processed_image, verbose=0)
    
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])
    
    return jsonify({
        'class': class_name[2:].strip(),
        'confidence': f"{confidence_score*100:.0f}%"
    })

if __name__ == '__main__':
    load_keras_model()
    app.run(debug=True)