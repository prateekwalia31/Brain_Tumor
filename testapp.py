
from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('cnn_brain_tumor1')

class_labels = [0, 1, 2, 3]

labels_str = {0:'Glioma', 1:'Meningioma', 2:'No Tumor', 3:'Pituitary'}

# Load the input image

img_path = '/Users/apple/Documents/Pycharm/Brain_Tumor/Testing Files/Pituitary/Te-pi_0018.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to (256, 256)
img_resized = cv2.resize(img, (256, 256))

# Reshape the image to (1, 256, 256, 1) to match the input shape of the model
img_input = img_resized.reshape(1, 256, 256, 1)

# Normalize the input image
img_input = img_input / 255.0

# Predict the class probabilities
probs = model.predict(img_input)
# Get the predicted class label
predicted_class = class_labels[np.argmax(probs)]

print('Predicted class:', labels_str[predicted_class])

#print('Class probabilities:', probs)