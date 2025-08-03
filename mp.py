import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Medicinal Plant Classifier", layout="centered")

# Download and Load Model
@st.cache_resource
def download_model():
    model_file = "medicinal_plant_model.h5"
    if not os.path.exists(model_file):
        url = "https://drive.google.com/uc?id=1kFhEUUcOICRVMHI-nZQL0VKRrD5L0UGh"
        gdown.download(url, model_file, quiet=False)
    return model_file

model_path = download_model()
model = load_model(model_path)

# Class labels
classes = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avocado', 'Bamboo',
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka',
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine',
    'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni',
    'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

# Medicinal plant information (fill in your data as needed)
plant_info = {
    'Aloevera': {
        'medicinal_properties': 'Aloe vera is anti-inflammatory and rich in antioxidants.',
        'used_for': 'Used for treating burns, skin irritation, and digestive issues.',
        'how_to_use': 'Apply gel topically or consume juice in moderation.'
    },
    'Tulasi': {
        'medicinal_properties': 'Tulasi (Holy Basil) has antibacterial and antiviral properties.',
        'used_for': 'Boosts immunity, treats coughs and colds.',
        'how_to_use': 'Use leaves in tea or chew raw leaves daily.'
    },
    # Add the rest of the plants similarly...
}

# App Header
st.title("🌿 Medicinal Plant Leaf Classifier")
st.write("Upload a leaf image to identify the plant and explore its medicinal benefits.")

# File Upload
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=300)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    prediction = classes[class_index]

    # Look up info
    info = plant_info.get(prediction, {
        'medicinal_properties': 'Information not available.',
        'used_for': 'Information not available.',
        'how_to_use': 'Information not available.'
    })

    # Output Results
    st.success(f"🌱 Identified Plant: **{prediction}**")

    st.markdown("### 🧪 Medicinal Properties")
    st.info(info["medicinal_properties"])

    st.markdown("### 💊 Common Uses")
    st.write(info["used_for"])

    st.markdown("### 🧉 How to Use")
    st.write(info["how_to_use"])
