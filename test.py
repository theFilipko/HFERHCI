import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline

train_data = "emotion\\train"
test_data = "emotion\\test"

def one_hot_label(img):
    label = img.split('.')[0]
    if label == "happy":
        ohl = np.array([1, 0])
    elif label == "sad":
        ohl = np.array([0, 1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 128, 128, 1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 128, 128, 1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape=[128, 128, 1]))
model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=5, padding='same'))

model.add(keras.layers.Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=5, padding='same'))

model.add(keras.layers.Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=5, padding='same'))

model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Dense(2, activation='softmax'))
optimizer = keras.optimizers.Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=60, batch_size=100)
model.summary()



fig = plt.figure(figsize=(14, 14))
for cnt, data in enumerate(testing_images[:30]):
    y = fig.add_subplot(6, 5, cnt+1)
    img = data[0]
    data = img.reshape(1, 128, 128, 1)
    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label = 'sad'
    else:
        str_label = 'happy'

    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

fig.show()
plt.show()
