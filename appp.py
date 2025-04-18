import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

# Define class labels and medication suggestions
class_labels = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
    3: "No Tumor"
}

medications = {
    "Glioma": "Temozolomide (chemotherapy), radiation therapy, and surgery.",
    "Meningioma": "Surgical resection, stereotactic radiosurgery, or observation.",
    "Pituitary": "Hormone therapy (e.g., bromocriptine), surgery, or radiation.",
    "No Tumor": "No medication required. Maintain regular check-ups."
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Batch dimension
    return image

# Streamlit App UI
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI brain scan image to classify the tumor type and get recommended medications.")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and predict
    if st.button("Classify and Get Medication"):
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            tumor_type = class_labels[predicted_class]
            medication = medications[tumor_type]

        st.success(f"**Prediction:** {tumor_type}")
        st.info(f"**Recommended Medication:** {medication}")
