import os
import cv2
import csv
import random
from shutil import copyfile


def number_to_emotion(number):
    if number == '1.0':
        return "anger"
    if number == '2.0':
        return "contempt"
    if number == '3.0':
        return "disgust"
    if number == '4.0':
        return "fear"
    if number == '5.0':
        return "joy"
    if number == '6.0':
        return "neutral"
    if number == '7.0':
        return "sadness"
    if number == '8.0':
        return "surprise"


def extract_person(id):
    org = "C:\\Users\\teleu\\PycharmProjects\\WtfEmotions\\data\\"
    dest = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\" + id

    # get results (dataset)
    with open(org + 'results\\dataset_' + id + '.csv') as f:
        reader = csv.reader(f, delimiter=',')
        data = [i for i in reader]
    data = data[1:]

    i = 0
    for datum in data:
        image = cv2.imread(org + datum[0])
        cv2.imwrite(dest + "\\" + number_to_emotion(datum[1]) + "." + str(i) + ".jpg", image)
        i += 1


def normalise_person_dataset(id=None, img_per_emotion=80):
    org = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\{}\\data".format(id)
    dest = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\{}\\data_norm".format(id)

    # get the number of emotions and the number of images for each emotion
    files = os.listdir(org)
    emotions = {}
    for file in files:
        emotion = file.split('.')[0]
        if emotion in emotions:
            emotions[emotion] += 1
        else:
            emotions[emotion] = 1

    # duplicate random images of the emotion to match the required number of images per emotion
    for emotion, count in emotions.items():
        # get all files for the emotion
        names = []
        for file in files:
            if emotion in file:
                names.append(file)
        # duplicate random images
        while len(names) < img_per_emotion:
            names.append(random.choice(names))
        # save the images to the new file
        for i, name in enumerate(names):
            cv2.imwrite(os.path.join(dest, emotion + "." + str(i) + ".jpg"), cv2.imread(os.path.join(org, name)))


def split_person_dataset(id=None, testing_per_emotion=20):
    org = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\{}\\data_norm".format(id)
    dest_test = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\{}\\test".format(id)
    dest_train = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\{}\\train".format(id)

    files = os.listdir(org)
    emotions = {}
    for file in files:
        emotion = file.split('.')[0]
        if emotion in emotions:
            emotions[emotion].append(file)
        else:
            emotions[emotion] = [file]

    for emotion, names in emotions.items():
        testing_images = random.sample(names, testing_per_emotion)
        for name in testing_images:
            copyfile(os.path.join(org, name), os.path.join(dest_test, name))

        training_images = [img for img in names if img not in testing_images]
        for name in training_images:
            copyfile(os.path.join(org, name), os.path.join(dest_train, name))


if __name__ == "__main__":
    #normalise_person_dataset(id="f")
    split_person_dataset(id="f")
