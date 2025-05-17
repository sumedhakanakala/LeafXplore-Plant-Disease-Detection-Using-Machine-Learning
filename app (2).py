import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Define class names and corresponding plant names
class_map = {
    0: ("Tomato", "Healthy"),
    1: ("Tomato", "Powdery"),
    2: ("Tomato", "Rust")
}

# Styling
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100

    plant, condition = class_map[predicted_index]
    st.markdown(f"<div class='prediction'>🌱 **Prediction:** {plant} - {condition} <br> 🧠 **Confidence:** {confidence:.2f}%</div>", unsafe_allow_html=True)
