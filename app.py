import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os


# Function to load categories from the category.txt file
def load_categories(file_path):
    categories = {}
    with open(file_path, 'r') as f:
        next(f)  # Skip the header line (id and name)
        for line in f:
            try:
                class_id, class_name = line.strip().split('\t')  # Split by tab
                categories[int(class_id)] = class_name
            except ValueError:
                st.warning(f"Skipping malformed line: {line.strip()}")
    return categories


# Load categories
categories = load_categories(r"C:\Users\arzua\Downloads\food\food — копия\category.txt")

# Streamlit app title
st.title("Food Recognition App")

# Load pre-trained model
MODEL_PATH = r"C:\Users\arzua\Downloads\food\food — копия\image_classification_model.h5"
model = load_model(MODEL_PATH)
st.write("Model loaded successfully.")

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to the required input size
    img_array = img_to_array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Get the class name
    predicted_food = categories.get(predicted_class + 1, "Unknown food")  # +1 to match category IDs

    # Display results
    st.write(f"**Predicted Class:** {predicted_food}")
    st.write(f"**Confidence:** {confidence:.2f}")



# cd "C:\Users\arzua\Downloads\food\food — копия"
# streamlit run app.py