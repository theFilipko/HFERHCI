# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm


train_data = "emotion\\train"
test_data = "emotion\\test"

def one_hot_label(img):
    label = img.split('.')[0]
    if label == "happy":
        ohl = 0
    elif label == "sad":
        ohl = 1
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 340))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 340))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images

training_data = train_data_with_label()
testing_data = test_data_with_label()

# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, we divide the values by 255.
# It's important that the training set and the testing set are preprocessed in the same way
train_images = np.array([i[0] for i in training_data]).reshape(-1,512,340,1)
train_labels = np.array([i[1] for i in training_data])
test_images = np.array([i[0] for i in testing_data]).reshape(-1,512,340,1)
test_labels = np.array([i[1] for i in testing_data])

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(512, 340, 1)),

    keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, padding='same'),

    keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, padding='same'),

    keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, padding='same'),

    #keras.layers.Conv2D(filters=130, kernel_size=5, strides=2, padding='same', activation='relu'),
    #keras.layers.MaxPool2D(pool_size=5, padding='same'),

    keras.layers.Dropout(rate=0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, batch_size=100)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
