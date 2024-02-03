import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

# Function to load the Keras model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('dog_cat_detector_model_Final_1(2).h5')

# Load the pre-trained model
model = load_model()

st.title('Dog or Cat Predictor')
st.write("""
## 1️⃣ About
""")
st.write("""
Hi all, Welcome to this project. It is a Cat Or Dog Recognizer App.
	""")
st.write("""
You have to upload your own image here:
	""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.write("""
    Preview 👀 Of Given Image:
    """)
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    # Display the uploaded image using st.image()
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Convert image to RGB (if it's not already in RGB format)
    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize and preprocess the image
    img_resized = cv2.resize(img_array, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Make prediction
    prediction = model.predict(img_resized)
    if prediction[0][0] > 0.5:
        st.write("## Model predicts it as an image of a DOG 🐶")
    else:
        st.write("## Model predicts it as an image of a CAT 🐱")

#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")