import tensorflow as tf
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image

# === 1. LOAD TRAINED MODEL ===
model_path = "skin_cancer_model.h5"
model = load_model(model_path)
print("✅ Model Loaded Successfully!")

# === 2. DEFINE CLASS NAMES (EXTRACTED FROM TRAINING DATA) ===
class_names = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 
               'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 
               'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesions']

# === 3. IMAGE PREPROCESSING FUNCTION ===
IMG_SIZE = (160, 160)  # Match model input size

def preprocess_image(image):
    image = image.convert('RGB')  # Ensure RGB format
    image = image.resize(IMG_SIZE)  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# === 4. PREDICTION FUNCTION ===
def predict_skin_cancer(image):
    image = preprocess_image(image)
    predictions = model.predict(image)[0]  # Get model predictions
    result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return result

# === 5. CREATE GRADIO LIVE DEMO ===
interface = gr.Interface(
    fn=predict_skin_cancer,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🩺 AI-Powered Skin Cancer Detection",
    description="Upload an image of a skin lesion and get a classification result."
)

# === 6. RUN LIVE DEMO ===
interface.launch(share=True)