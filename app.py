from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

app = Flask(__name__)

# Load the trained model
model = load_model('medicinal_plant_model.h5')

# Define the class labels (40 classes)
classes = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avocado', 'Bamboo', 
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 
    'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 
    'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)
            class_index = np.argmax(preds[0])
            print("Predicted class index:", class_index)

            # Ensure index is valid
            if 0 <= class_index < len(classes):
                prediction = classes[class_index]
            else:
                prediction = "Unknown class"

            return render_template('index.html', prediction=prediction, image_path=img_path)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

app = Flask(__name__)

# Load the trained model
model = load_model('medicinal_plant_model.h5')

# Define the class labels (40 classes)
classes = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avocado', 'Bamboo', 
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 
    'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 
    'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)
            class_index = np.argmax(preds[0])
            print("Predicted class index:", class_index)

            # Ensure index is valid
            if 0 <= class_index < len(classes):
                prediction = classes[class_index]
            else:
                prediction = "Unknown class"

            return render_template('index.html', prediction=prediction, image_path=img_path)

    return render_template('index.html', prediction=prediction)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

