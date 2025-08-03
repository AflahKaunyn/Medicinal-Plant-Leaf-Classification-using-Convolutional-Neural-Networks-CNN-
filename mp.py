import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
from PIL import Image
import os

# Ensure the model is downloaded and loaded only once
@st.cache_resource
def get_model():
    url = "https://drive.google.com/uc?id=1kFhEUUcOICRVMHI-nZQL0VKRrD5L0UGh"
    output = "medicinal_plant_model.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output)
    return model

# Load the model
model = get_model()

# Class labels
classes = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avocado', 'Bamboo',
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka',
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine',
    'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni',
    'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

plant_info = {
      'Aloevera': {
        'medicinal_properties': 'Anti-inflammatory, soothing, healing.',
        'used_for': 'Burns, skin hydration, digestion.',
        'how_to_use': 'Apply gel on skin or drink juice.'
    },
    'Amla': {
        'medicinal_properties': 'Rich in Vitamin C, antioxidant, immunity booster.',
        'used_for': 'Immunity, digestion, hair care.',
        'how_to_use': 'Consume raw or as juice/powder.'
    },
    'Amruta_Balli': {
        'medicinal_properties': 'Anti-diabetic, immune booster, detoxifying.',
        'used_for': 'Diabetes, fever, respiratory problems.',
        'how_to_use': 'Boil stems in water, consume as decoction.'
    },
    'Arali': {
        'medicinal_properties': 'Antimicrobial, anti-inflammatory.',
        'used_for': 'Skin issues, respiratory problems.',
        'how_to_use': 'Used in traditional formulations, consult practitioner.'
    },
    'Ashoka': {
        'medicinal_properties': 'Uterine tonic, anti-inflammatory.',
        'used_for': 'Menstrual disorders, internal bleeding.',
        'how_to_use': 'Use bark powder or decoction under guidance.'
    },
    'Ashwagandha': {
        'medicinal_properties': 'Adaptogenic, anti-stress, rejuvenator.',
        'used_for': 'Stress, weakness, low stamina.',
        'how_to_use': 'Consume root powder with warm milk or water.'
    },
    'Avocado': {
        'medicinal_properties': 'Rich in healthy fats, antioxidant, anti-inflammatory.',
        'used_for': 'Heart health, skin nourishment.',
        'how_to_use': 'Eat raw fruit or apply mashed pulp to skin.'
    },
    'Bamboo': {
        'medicinal_properties': 'Anti-inflammatory, bone strengthener.',
        'used_for': 'Joint pain, respiratory issues.',
        'how_to_use': 'Juice or decoction of bamboo shoots/leaves.'
    },
    'Basale': {
        'medicinal_properties': 'Cooling, anti-ulcer, anti-inflammatory.',
        'used_for': 'Acidity, skin boils, ulcers.',
        'how_to_use': 'Use leaves in food or as poultice.'
    },
    'Betel': {
        'medicinal_properties': 'Antimicrobial, digestive aid.',
        'used_for': 'Cough, digestion, mouth ulcers.',
        'how_to_use': 'Chew leaves or make herbal infusion.'
    },
    'Betel_Nut': {
        'medicinal_properties': 'Stimulant, astringent, digestive.',
        'used_for': 'Indigestion, oral health.',
        'how_to_use': 'Dried and chewed (in moderation).'
    },
    'Brahmi': {
        'medicinal_properties': 'Brain tonic, anti-anxiety, antioxidant.',
        'used_for': 'Memory enhancement, stress relief.',
        'how_to_use': 'Consume extract or brew as tea.'
    },
    'Castor': {
        'medicinal_properties': 'Laxative, anti-inflammatory.',
        'used_for': 'Constipation, joint pain.',
        'how_to_use': 'Use oil externally or consume under guidance.'
    },
    'Curry_Leaf': {
        'medicinal_properties': 'Antioxidant, hair tonic, digestive.',
        'used_for': 'Hair health, diabetes, indigestion.',
        'how_to_use': 'Add fresh leaves to food or consume powder.'
    },
    'Doddapatre': {
        'medicinal_properties': 'Expectorant, antimicrobial, carminative.',
        'used_for': 'Cough, cold, indigestion.',
        'how_to_use': 'Crush and use juice or boil in tea.'
    },
    'Ekka': {
        'medicinal_properties': 'Cardiotonic, anti-inflammatory.',
        'used_for': 'Wound healing, skin diseases.',
        'how_to_use': 'Apply leaf paste externally or use decoction.'
    },
    'Ganike': {
        'medicinal_properties': 'Digestive, antimicrobial.',
        'used_for': 'Stomach issues, skin problems.',
        'how_to_use': 'Use leaf extract or decoction.'
    },
    'Gauva': {
        'medicinal_properties': 'Antioxidant, antimicrobial, rich in Vitamin C.',
        'used_for': 'Diarrhea, cold, immunity.',
        'how_to_use': 'Eat ripe fruit or boil leaves for tea.'
    },
    'Geranium': {
        'medicinal_properties': 'Astringent, anti-inflammatory, aromatic.',
        'used_for': 'Skin care, mood lifting.',
        'how_to_use': 'Use essential oil or leaf infusion.'
    },
    'Henna': {
        'medicinal_properties': 'Cooling, antifungal, astringent.',
        'used_for': 'Hair dye, scalp health, wounds.',
        'how_to_use': 'Apply paste to hair or skin.'
    },
    'Hibiscus': {
        'medicinal_properties': 'Antioxidant, hair tonic, cooling.',
        'used_for': 'Hair fall, blood pressure.',
        'how_to_use': 'Use flowers in tea or apply paste to hair.'
    },
    'Honge': {
        'medicinal_properties': 'Antibacterial, insect repellent.',
        'used_for': 'Skin conditions, pest control.',
        'how_to_use': 'Use oil or leaf extract externally.'
    },
    'Insulin': {
        'medicinal_properties': 'Anti-diabetic, blood sugar regulator.',
        'used_for': 'Diabetes management.',
        'how_to_use': 'Chew fresh leaves or prepare tea.'
    },
    'Jasmine': {
        'medicinal_properties': 'Antidepressant, aphrodisiac, calming.',
        'used_for': 'Stress, skin care.',
        'how_to_use': 'Use flowers in oil or as tea.'
    },
    'Lemon': {
        'medicinal_properties': 'Rich in Vitamin C, detoxifier.',
        'used_for': 'Sore throat, digestion, skin brightening.',
        'how_to_use': 'Add juice to warm water or apply topically.'
    },
    'Lemon_grass': {
        'medicinal_properties': 'Digestive, antimicrobial, calming.',
        'used_for': 'Fever, anxiety, stomach ache.',
        'how_to_use': 'Use in tea or as essential oil.'
    },
    'Mango': {
        'medicinal_properties': 'Antioxidant, digestive.',
        'used_for': 'Anemia, immunity, digestion.',
        'how_to_use': 'Eat ripe fruit or use leaf decoction.'
    },
    'Mint': {
        'medicinal_properties': 'Cooling, antispasmodic, digestive aid.',
        'used_for': 'Indigestion, headache, nausea.',
        'how_to_use': 'Use fresh leaves in tea or chew raw.'
    },
    'Nagadali': {
        'medicinal_properties': 'Astringent, hemostatic.',
        'used_for': 'Wounds, cuts.',
        'how_to_use': 'Apply leaf paste on affected area.'
    },
    'Neem': {
        'medicinal_properties': 'Antibacterial, antifungal, blood purifier.',
        'used_for': 'Skin conditions, blood detox, oral hygiene.',
        'how_to_use': 'Apply paste on skin or drink neem water.'
    },
    'Nithyapushpa': {
        'medicinal_properties': 'Anti-inflammatory, hemostatic.',
        'used_for': 'Wounds, piles.',
        'how_to_use': 'Apply paste externally or take as decoction.'
    },
    'Nooni': {
        'medicinal_properties': 'Anti-inflammatory, immune booster.',
        'used_for': 'Pain relief, infection, fatigue.',
        'how_to_use': 'Drink fruit juice or use leaf decoction.'
    },
    'Pappaya': {
        'medicinal_properties': 'Digestive enzymes, antioxidant.',
        'used_for': 'Indigestion, skin health.',
        'how_to_use': 'Eat ripe fruit or apply paste.'
    },
    'Pepper': {
        'medicinal_properties': 'Stimulant, carminative, antioxidant.',
        'used_for': 'Cold, digestion, metabolism.',
        'how_to_use': 'Use in food or with honey for cough.'
    },
    'Pomegranate': {
        'medicinal_properties': 'Rich in iron, antioxidant, astringent.',
        'used_for': 'Anemia, diarrhea.',
        'how_to_use': 'Eat fruit or use rind decoction.'
    },
    'Raktachandini': {
        'medicinal_properties': 'Blood purifier, cooling.',
        'used_for': 'Skin diseases, heat ailments.',
        'how_to_use': 'Use bark powder in decoction.'
    },
    'Rose': {
        'medicinal_properties': 'Cooling, anti-inflammatory, aromatic.',
        'used_for': 'Skin care, mood enhancer.',
        'how_to_use': 'Use rose water or petals in tea.'
    },
    'Sapota': {
        'medicinal_properties': 'Energy booster, anti-inflammatory.',
        'used_for': 'Diarrhea, fatigue.',
        'how_to_use': 'Eat ripe fruit.'
    },
    'Tulasi': {
        'medicinal_properties': 'Anti-inflammatory, antimicrobial, antioxidant.',
        'used_for': 'Common cold, respiratory issues, stress.',
        'how_to_use': 'Boil leaves in water and drink as tea or chew fresh leaves.'
    },
    'Wood_sorel': {
        'medicinal_properties': 'Cooling, digestive, febrifuge.',
        'used_for': 'Fever, diarrhea, scurvy.',
        'how_to_use': 'Crush and take juice or use in salads.'
    }
}

st.set_page_config(page_title="Medicinal Plant Classifier", layout="centered")

st.title("🌿 Medicinal Plant Leaf Classifier")
st.write("Upload a leaf image to identify the plant and explore its medicinal benefits.")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=300)

    # Preprocess image for model
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    prediction = classes[class_index]

    # Plant info
    info = plant_info.get(prediction, {
        'medicinal_properties': 'Information not available.',
        'used_for': 'Information not available.',
        'how_to_use': 'Information not available.'
    })

    st.success(f"🌱 Identified Plant: **{prediction}**")
    st.markdown("### 🧪 Medicinal Properties")
    st.info(info['medicinal_properties'])
    st.markdown("### 💊 Common Uses")
    st.write(info['used_for'])
    st.markdown("### 🧉 How to Use")
    st.write(info['how_to_use'])

# Optional: If no file is uploaded, prompt the user
else:
    st.info("Please upload a leaf image to begin.")
