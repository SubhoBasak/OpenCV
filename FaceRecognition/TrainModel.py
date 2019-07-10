import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

recognizer = cv2.face.LBPHFaceRecognizer_create()

images = []
labels = []

files = os.listdir('DatasetImages/')

for i in files:
    img = cv2.imread('DatasetImages/'+i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)
    labels.append(i.split('-')[0])
    cv2.imshow('Processing...', img)
    cv2.waitKey(10)

images = np.array(images)
labels = np.array(labels)

name_encoder = LabelEncoder()
labels = name_encoder.fit_transform(labels)

with open('TrainningFiles/Labels.txt', 'w') as file:
    for i in name_encoder.classes_:
        file.write(i+'\n')
    file.close()

recognizer.train(images, labels)
recognizer.save('TrainningFiles/Trainner.yml')

cv2.destroyAllWindows()
