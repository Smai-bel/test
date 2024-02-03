# app.py

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import urllib.request
from io import BytesIO
from PIL import Image

# Function to download the model from GitHub
def download_model_from_github(github_url):
    response = urllib.request.urlopen(github_url)
    model_bytes = BytesIO(response.read())
    return model_bytes

# Download the model from GitHub
github_model_url = "https://github.com/Smai-bel/test/main/dog_cat_detector_model_Final_1(2).h5"  # Replace with the actual URL
model_bytes = download_model_from_github(github_model_url)



# Function to preprocess the image
def preprocess_image(image_path):
    pil_image = Image.open(image_path)
    pil_image_resized = pil_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Main Streamlit code
st.title("Dog vs Cat Image Classifier")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the resized image for the model
        img_array = preprocess_image(uploaded_file)

        # Load the model from BytesIO
        loaded_model = load_model(model_bytes)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Print raw predictions
        st.write("Raw Predictions:", predictions[0].tolist())

        # Manually interpret predictions based on your model's output with a threshold
        threshold = 0.5
        predicted_class = 'Dog' if predictions[0][0] >= threshold else 'Cat'
        confidence = predictions[0][0]

        # Display the result
        st.subheader("Prediction:")
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {float(confidence):.2%}")

    except Image.UnidentifiedImageError as e:
        st.error(f"Error: Unable to identify image file. Please upload a valid image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

