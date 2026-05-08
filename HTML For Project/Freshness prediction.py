import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= CONFIG =================

MODEL_PATH = r"F:\dd\archive\Research_Results\EfficientNet_Merged\EfficientNet_Final.keras"
IMAGE_PATH = r"F:\Testing apples\fruit_652_2.jpg"  # <-- Change this

IMG_SIZE = 224

# ================= LOAD MODEL =================

print("Loading model...")
model = load_model(MODEL_PATH)

# ================= CLASS NAMES =================
# IMPORTANT: Must match training order

class_names = ['Blotch_Apple', 'Normal_Apple', 'Rot_Apple', 'Scab_Apple']

# ================= PREPROCESS IMAGE =================

print("Processing image...")

img = load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# ================= PREDICT =================

prediction = model(img_array, training=False)
prediction = prediction.numpy()
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

print("\n==============================")
print("Predicted Class :", class_names[predicted_class])
print("==============================")
