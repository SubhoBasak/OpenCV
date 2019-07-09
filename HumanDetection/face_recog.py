import os
import cv2
import numpy as np

##########
video = cv2.VideoCapture('videoplayback.mp4')

face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_profileface.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_lowerbody.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

##########
#count = 0
#images = os.listdir('images')
#for i in range(len(images)):
#    images[i] = 'images/'+images[i]

while True:
#########
#    frame = cv2.imread(images[count])
#    count += 1
#    if count > len(images)-1:
#        count = 0
#########
    rate, frame = video.read()

    Y, X, Z = frame.shape
    frame = cv2.resize(frame, (100, 100))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    profiles = profile_cascade.detectMultiScale(gray, 1.3, 1)
    fullbodies = fullbody_cascade.detectMultiScale(gray, 1.1, 1)
    upperbodies = upperbody_cascade.detectMultiScale(gray, 1.4, 1)
    lowerbodies = lowerbody_cascade.detectMultiScale(gray, 1.5, 1)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [255, 0, 0], 2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, w*h/100, 1)
        smiles = smile_cascade.detectMultiScale(roi_gray, w*h/100, 1)
        for xx, yy, ww, hh in eyes:
            if yy <= h/2+y+(h*30/100):
                cv2.rectangle(frame, (x+xx, y+yy), (x+xx+ww, y+yy+hh), [123, 123, 22], 2)
        for xx, yy, ww, hh in smiles:
            if yy >= h/2+y-(h*20/100):
                cv2.rectangle(frame, (x+xx, y+yy), (x+xx+ww, y+yy+hh), [10, 23, 19], 2)
                break
    for x, y, w, h in profiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 0], 2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, w*h/100, 1)
        smiles = smile_cascade.detectMultiScale(roi_gray, w*h/100, 1)
        for xx, yy, ww, hh in eyes:
            if yy <= h/2+y+(h*30/100):
                cv2.rectangle(frame, (x+xx, y+yy), (x+xx+ww, y+yy+hh), [12, 223, 22], 2)
        for xx, yy, ww, hh in smiles:
            if yy >= h/2+y-(h*20/100):
                cv2.rectangle(frame, (x+xx, y+yy), (x+xx+ww, y+yy+hh), [212, 23, 99], 2)
                break
    for x, y, w, h in fullbodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 0, 255], 2)
    for x, y, w, h in upperbodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [255, 255, 0], 2)
    for x, y, w, h in lowerbodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 255], 2)
    cv2.imshow('output', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#video.release()
cv2.destroyAllWindows()
