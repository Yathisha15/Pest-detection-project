# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
import numpy as np

# Load the pre-trained model
model = load_model("model/v4_pred_cott_dis2.h5")
print('@@ Model loaded')

def predict_cotton_disease(image_path):
    try:
        # Load and preprocess the image
        test_image = load_img(image_path, target_size=(150, 150))
        test_image = img_to_array(test_image) / 255.0
        test_image = test_image.reshape((1, 150, 150, 3))

        # Make predictions
        result = model.predict(test_image).round(3)
        print('@@ Raw result = ', result)

        # Get the predicted class
        pred_class = int(np.argmax(result))

        # Define the classes and corresponding templates
        classes = ["Healthy Cotton Plant", "Diseased Cotton Plant", "Healthy Cotton Plant"]
        templates = ["healthy_plant_leaf.html", "disease_plant.html", "healthy_plant.html"]

        return classes[pred_class], templates[pred_class]

    except Exception as e:
        print('Error during prediction: ', str(e))
        return "Error during prediction", 'error_page.html'

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the uploaded image file
            file = request.files['image']
            if file:
                # Save the file
                file_path = os.path.join('static/user_uploaded', file.filename)
                file.save(file_path)

                # Perform prediction
                pred, output_page = predict_cotton_disease(image_path=file_path)

                # Render the result template
                return render_template(output_page, pred_output=pred, user_image=file_path)
            else:
                return render_template('error_page.html', error='No file uploaded.')

        except Exception as e:
            print('Error during request processing: ', str(e))
            return render_template('error_page.html', error='Error during request processing.')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)