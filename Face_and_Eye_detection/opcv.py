import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video = cv2.VideoCapture('videoplayback.mp4')

while True:
    rate, frame = video.read()
    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 1)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [255, 0, 0], 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eye = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eye:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), [0, 255, 255], 1)
    cv2.imshow('output', frame)
    cv2.waitKey(2)
