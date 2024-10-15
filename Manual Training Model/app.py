from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from keras import layers,models
import os
import numpy as np
import cv2
import random


ruta_train = 'train/'
ruta_predict = 'valid/0/myimage.jpg'
labels = os.listdir(ruta_train)
width = 300
height = 300
model = models.load_model('mimodelo.keras', compile=False)
# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to the homepage where the user can upload an image
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle image upload
@app.route('/', methods=['POST'])
def upload_image():
    # Check if a file has been uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if the user has selected a file
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image and get the prediction
        grupo, porcentaje = process_image(filepath)
        
        # Redirect to show the uploaded image, its details, and prediction
        return redirect(url_for('display_image', filename=filename, grupo=grupo, porcentaje=porcentaje))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


# A function to process the image with Python
def process_image(image_path):
    # Open the image using PIL
    with Image.open(image_path) as img:
        # Convert the PIL image to a NumPy array
        img = np.array(img)

        img = cv2.resize(img, (width, height))

        result = model.predict(np.array([img]))[0]

        porcentaje = max(result)*100

        grupo = labels[result.argmax()]
        print(grupo, round(porcentaje))
        return grupo, round(porcentaje)


# Route to display the uploaded image and its size
@app.route('/display/<filename>/<grupo>/<int:porcentaje>')
def display_image(filename, grupo, porcentaje):
    return render_template('display.html', filename=filename, grupo=grupo, porcentaje=porcentaje)


# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# The Flask app entry point
if __name__ == "__main__":
    app.run(debug=True)
