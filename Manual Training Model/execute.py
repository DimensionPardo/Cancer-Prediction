import tensorflow as tf
from keras import layers,models
import os
import numpy as np
import cv2
import random
ruta_train = 'train/'

def get_random_file_from_folder(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Check if the folder is empty
    if not files:
        return None
    
    # Select a random file
    random_file = random.choice(files)
    
    return os.path.join(folder_path, random_file)

# Example usage
folder_path = '/path/to/your/folder'  # Replace with your folder path
ruta_predict = get_random_file_from_folder('valid/1/')


labels = os.listdir(ruta_train)
width = 640
height = 640

model = models.load_model('mimodelo.keras', compile=False)


my_image = cv2.imread("valid/0/625_1097006067_png.rf.74100fa7cf2bf518b73b7530fbfc409f.jpg")
my_image = cv2.resize(my_image, (width, height))

result = model.predict(np.array([my_image]))

print(result)

