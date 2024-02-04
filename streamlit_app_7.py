import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
import tempfile

# Function to preprocess the image
def preprocess_image(image_path):
    pil_image = Image.open(image_path)
    pil_image_resized = pil_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Replace this with your actual Google Drive direct download link
model_link = "https://drive.google.com/uc?export=download&id=1DssTM7IZ_porxkdU0Ickfxh3e8b1N4j_"

# Download the model from the provided link
response = requests.get(model_link)
model_bytes = BytesIO(response.content)

# Save BytesIO content to a temporary file
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(model_bytes.getvalue())
temp_file_path = temp_file.name
temp_file.close()

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

        # Load the model from the temporary file
        model = tf.keras.models.load_model(temp_file_path)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Manually interpret predictions based on your model's output with a threshold
        threshold = 0.5
        predicted_class = 'Dog' if predictions[0][0] >= threshold else 'Cat'
        confidence = predictions[0][0]

        # Display the result
        st.subheader("Prediction:")
        st.write(f"this picture corespond to a : {predicted_class}")

        if predicted_class == 'Cat':
            confidence = 1.0 - confidence  # Invert confidence for cats
        st.write(f"we are Confident at: {float(confidence):.2%}")

    except Image.UnidentifiedImageError as e:
        st.error(f"Error: Unable to identify the image file. Please upload a valid image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Remove the temporary file after using it
import os
os.remove(temp_file_path)
