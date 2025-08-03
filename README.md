# Medicinal Plant Leaf Classification using CNN

This project is a deep learning-based web application that classifies medicinal plants from their leaf images using a Convolutional Neural Network (CNN). Once a leaf image is uploaded, the system predicts the plant species and displays its medicinal properties, common uses, and usage instructions.

## 🌿 Project Highlights

- Image classification using CNN trained on a custom dataset of medicinal plant leaves.
- Flask-based web interface for user-friendly interaction.
- Supports 40+ medicinal plant species, including Tulasi, Neem, Aloe Vera, Ashwagandha, and more.
- Provides details like:
  - Medicinal properties
  - Common health benefits
  - Instructions for safe usage

## 🧠 Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: Leaf image (RGB, resized to fixed dimensions)
- Output: Plant species (Multi-class classification)
- Training: Trained on labeled dataset of medicinal leaf images
- Accuracy: Achieved high accuracy on validation and test sets

## 🖼️ Dataset

- Includes images for 40 classes of medicinal plants.
- Each class has 100–200 labeled images.
- Data Augmentation techniques were used to improve generalization.
- Amruta_Balli, Amla, Aloevera, Arali, Ashoka, Ashwagandha, Avocado, Bamboo, Basale, Betel, Betel_Nut, Brahmi, Castor, Curry_Leaf, Doddapatre, Ekka, Ganike, Geranium, Guava, Henna, Hibiscus, Honge, Insulin, Jasmine, Lemon, Lemon_grass, Mango, Mint, Nagadali, Neem, Nithyapushpa, Nooni, Pappaya, Pepper, Pomegranate, Raktachandini, Rose, Sapota, Tulasi, Wood_sorel


## 💻 Technologies Used

- Python
- Flask
- TensorFlow / Keras or PyTorch (depending on your implementation)
- OpenCV (for image preprocessing)
- HTML/CSS + Bootstrap (for UI)

## 🚀 How to Run

1. **Clone the repository:**
   git clone https://github.com/yourusername/medicinal-plant-classification.git
   cd medicinal-plant-classification
   
2. **Set up environment:**
   pip install -r requirements.txt

3. **Run the Flask app:**
   python app.py


**📜 License** 

This project is intended for educational and research purposes only.  
Feel free to use or modify the code for non-commercial use with proper credit.  
No warranties provided.

