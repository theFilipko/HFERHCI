import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm


def number_to_emotion(number):
    if number == 1:
        return "anger"
    if number == 2:
        return "contempt"
    if number == 3:
        return "disgust"
    if number == 4:
        return "fear"
    if number == 5:
        return "joy"
    if number == 6:
        return "neutral"
    if number == 7:
        return "sadness"
    if number == 8:
        return "surprise"


def emotion_to_number(emotion):
    if emotion == "anger":
        return 1
    if emotion == "contempt":
        return 2
    if emotion == "disgust":
        return 3
    if emotion == "fear":
        return 4
    if emotion == "joy":
        return 5
    if emotion == "neutral":
        return 6
    if emotion == "sadness":
        return 7
    if emotion == "surprise":
        return 8


def create_labels():
    files = os.listdir(TEST_PATH)
    emotions = {}
    i = 0
    for file in files:
        emotion = file.split('.')[0]
        if emotion not in emotions:
            emotions[emotion] = i
            i += 1
    return emotions


'''def one_hot_label(image):
    # return emotion_to_number(img.split('.')[0])
    label = image.split('.')[0]
    if label == "joy":
        ohl = 0
    elif label == "neutral":
        ohl = 1
    elif label == "surprise":
        ohl = 2
    return ohl'''


def one_hot_label(image=None, labels=None):
    return labels[image.split('.')[0]]


def train_data_with_label(labels=None):
    imgs = []
    for i in tqdm(os.listdir(TRAIN_PATH)):
        path = os.path.join(TRAIN_PATH, i)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (IMG_X, IMG_Y))
        imgs.append([np.array(im), one_hot_label(image=i, labels=labels)])
    shuffle(imgs)
    return imgs


def test_data_with_label(labels=None):
    imgs = []
    for i in tqdm(os.listdir(TEST_PATH)):
        path = os.path.join(TEST_PATH, i)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (IMG_X, IMG_Y))
        imgs.append([np.array(im), one_hot_label(image=i, labels=labels)])
    shuffle(imgs)
    return imgs


if __name__ == "__main__":

    # loop for each person
    persons = ["c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "o", "x"]
    results = {}
    for person in persons:

        ID = person
        TRAIN_PATH = "data\\{}\\train".format(ID)
        TEST_PATH = "data\\{}\\test".format(ID)
        IMG_X = 124
        IMG_Y = 124

        # do it few times and collect the test accuracies
        accuracies = []
        for e in range(10):
            label_dic = create_labels()
            training_data = train_data_with_label(labels=label_dic)
            testing_data = test_data_with_label(labels=label_dic)

            train_images = np.array([i[0] for i in training_data]).reshape((-1, IMG_X, IMG_Y, 1))
            train_labels = np.array([i[1] for i in training_data])
            test_images = np.array([i[0] for i in testing_data]).reshape((-1, IMG_X, IMG_Y, 1))
            test_labels = np.array([i[1] for i in testing_data])

            # first model
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(IMG_X, IMG_Y, 1)),

                keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.2),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.2),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Dropout(rate=0.25),
                keras.layers.Flatten(),
                keras.layers.Dense(units=128, activation='relu'),
                keras.layers.Dropout(rate=0.5),
                #keras.layers.Dense(units=100, activation='relu'),
                #keras.layers.Dropout(rate=0.4),
                keras.layers.Dense(units=len(label_dic), activation='softmax')
            ])

            # second model
            """model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(IMG_X, IMG_Y, 1)),

                keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),

                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Flatten(),
                keras.layers.Dense(units=128, activation='softmax')
                # keras.layers.Dense(units=len(label_dic), activation='softmax')
            ])"""

            # third model
            """model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(IMG_X, IMG_Y, 1)),

                keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.MaxPool2D(pool_size=2, padding='same'),

                keras.layers.Flatten(),
                keras.layers.Dense(units=128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.Dense(units=len(label_dic) + 1, activation='softmax')
            ])"""

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_images, train_labels, epochs=30)

            test_loss, test_acc = model.evaluate(test_images, test_labels)
            accuracies.append((test_loss, test_acc))
            print('Test loss: {}\nTest accuracy: {}'.format(test_loss, test_acc))
            print("\nPerson: {}, {} of 10 \n".format(person, e))

            del model
            training_data.clear()
            testing_data.clear()

        results[person] = accuracies

        print("\nPerson: {}\n".format(person))
        for a in accuracies:
            print('Test accuracy: {}'.format(a[1]))

    print("\n")
    print("===========")
    print("= SUMMARY =")
    print("===========")
    print("\nCNN 2")
    for person, accuracies in results.items():
        print("\n\nPerson: {}\n".format(person))
        for a in accuracies:
            print('Test loss: {}\nTest accuracy: {}'.format(a[0], a[1]))

    # show some classification
    """fig = plt.figure(figsize=(14, 14))
    for cnt, data in enumerate(testing_data[5:25]):
        y = fig.add_subplot(6, 5, cnt+1)
        img = data[0]
        #data = img.reshape(-1, 124, 124, 1)
        model_out = model.predict(img.reshape((-1, IMG_X, IMG_Y, 1)))

        if np.argmax(model_out) == 0:
            str_label = "joy"
        elif np.argmax(model_out) == 1:
            str_label = "neutral"
        elif np.argmax(model_out) == 2:
            str_label = "surprise"

        if data[1] == 0:
            str_label_true = "joy"
        elif data[1] == 1:
            str_label_true = "neutral"
        elif data[1] == 2:
            str_label_true = "surprise"

        y.imshow(img)
        plt.title(str_label + " - " + str_label_true)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    fig.show()
    plt.show()"""

