# Dependencies
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
# Load the saved tf model from the saved_model directory

model = keras.models.load_model('cnn_brain_tumor1')

# class labels

class_labels = [0, 1, 2, 3]
labels_str = {0:'Glioma', 1:'Meningioma', 2:'No Tumor', 3:'Pituitary'}

@app.route('/request_api')
def request_api():
    return render_template('request_api.html')


@app.route('/')
def index():
    return "Page for Brain Tumor Detection."


@app.route('/classify_brain_tumor', methods=['POST'])
# @cross_orign()
def classify_brain_tumor():
    # get an image from the request (key=image)

    image_file = request.files['image']

    # Loading and preprocessing the image

    input_image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (256, 256))  # Reshape to 128x128
    input_image = input_image.reshape(1, 256, 256, 1)
    input_image = input_image.astype('float32') / 255.0  # Normalize

    # Classify the received image using the loaded model
    probs = model.predict(input_image)

    # Get the predicted class label
    predicted_class = class_labels[np.argmax(probs)]

    # Returning the predicted class as a JSON response
    response = jsonify({'text': 'The MRI Scan indicates: ' + labels_str[predicted_class]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    ''' Run the app'''
    app.run()
