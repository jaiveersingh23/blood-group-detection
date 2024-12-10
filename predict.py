# src/predict.py

"""import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the model
model = tf.keras.models.load_model('model_blood_group_detection.h5')
class_labels = sorted(os.listdir('dataset_blood_group'))

def predict_blood_group(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    image = Image.open(image_path).resize((256, 256))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)  # Preprocess for ResNet50
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    return predicted_class
"""
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

def predict_blood_group(image_path):
    try:
        # Load the trained model
        model = load_model("model_blood_group_detection.h5")  # Ensure the path is correct

        # Open and preprocess the image
        image = Image.open(image_path).resize((256, 256))  # Resize to match model input
        image_array = np.array(image)
        image_array = preprocess_input(image_array)  # Preprocess for ResNet50
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]  # Ensure this matches your model
        return classes[np.argmax(predictions)]
    except Exception as e:
        return f"Error during prediction: {e}"
