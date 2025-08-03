import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
from PIL import Image

# Download and load model
@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1kw9k-XBOCXGE2pHTCGhNeez-cZM3D9UW"
    output = "medicinal_plant_model1.h5"
    gdown.download(url, output, quiet=False)
    model = load_model(output)
    return model

model = download_model()

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

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224)).convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"Predicted class: {predicted_class}")



