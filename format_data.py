import os
import cv2
import csv


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


id = "x"
org = "C:\\Users\\teleu\\PycharmProjects\\WtfEmotions\\data\\"
dest = "C:\\Users\\teleu\\PycharmProjects\\HFERHCI\\data\\" + id

# get results (dataset)
n = os.listdir(org)
with open(org + 'results\\dataset_' + id + '.csv') as f:
    reader = csv.reader(f, delimiter=',')
    data = [i for i in reader]
data = data[1:]

i = 0
for datum in data:
    image = cv2.imread(org + datum[0])
    cv2.imwrite(dest + "\\" + number_to_emotion(datum[1]) + "." + str(i) + ".jpg", image)
    i += 1


"""images = []
for item in n:
    x = item.split('_')
    if "sad" in x:
        images.append(cv2.imread(org + "\\" + item))

i = 0
for img in images:
    cv2.imwrite(dest + "\\" + "sad." + str(i) + ".jpg", img)
    i += 1"""

# get results (dataset)
# check the number of classes (emotions) in the dataset
# get the images
# format them into folder structure (train, test, valid)
