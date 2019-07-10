import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('CascadeFiles/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('CascadeFiles/haarcascade_profileface.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainningFiles/Trainner.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

images = os.listdir('TestingImages')
for i in range(len(images)):
    images[i] = 'TestingImages/'+images[i]

file = open('TrainningFiles/Labels.txt', 'r')
labels = file.read()
labels = labels.split('\n')

count = 0
while True:
    img = cv2.imread(images[count])
    count += 1
    if count == len(images):
        count = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), [0, 255, 255], 2)
        enc_id, confidance = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.putText(img, labels[enc_id], (x, y), font, 0.4, [255, 255, 0])
        print(enc_id, confidance)
    for x, y, w, h in profiles:
        cv2.rectangle(img, (x, y), (x+w, y+h), [255, 0, 255], 2)
        enc_id, confidance = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.putText(img, labels[enc_id], (x, y), font, 0.4, [0, 255, 0])
        print(enc_id, confidance)
    cv2.imshow('OUTPUT', img)
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
