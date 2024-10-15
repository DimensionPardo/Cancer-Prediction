import tensorflow as tf
from keras import layers,models
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
import random

width = 300
height = 300
ruta_train = 'train/'
ruta_predict = 'valid/'

train_x = []
train_y = []

labels = os.listdir(ruta_train)

for i in os.listdir(ruta_train):
    for j in os.listdir(ruta_train + i):
        img = cv2.imread(ruta_train+i+'/'+j)
        resized_image = cv2.resize(img, (width, height))

        train_x.append(resized_image)

        for x,y in enumerate(labels):
            if y == i:
                array = np.zeros(len(labels))
                array[x]=1
                train_y.append(array)

x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3,3, input_shape=(width, height, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, 3,3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, 3,3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
    layers.Activation('sigmoid')
])



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

epochs = 50

model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'mimodelo.h5')