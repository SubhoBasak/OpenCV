import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('CascadeFiles/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('CascadeFiles/haarcascade_profileface.xml')

persons = os.listdir('TrainningImages')
for i in range(len(persons)):
    persons[i] = 'TrainningImages/'+persons[i]

for i in persons:
    images = os.listdir(i)
    for k, j in enumerate(images):
        img = cv2.imread(os.path.join(i, j))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        profiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), [0, 255, 255], 2)
            cv2.imwrite('DatasetImages/'+i[16:]+'-'+str(k)+'-.jpg', gray[int(y*4/5):int((y+h)*6/5), int(x*9/10):int((x+w)*11/10)])
        for x, y, w, h in profiles:
            cv2.rectangle(img, (x, y), (x+w, y+h), [255, 255, 0], 2)
            cv2.imwrite('DatasetImages/'+i[16:]+'-'+str(k*2)+'-.jpg', gray[int(y*4/5):int((y+h)*6/5), int(x*9/10):int((x+w)*11/10)])
        cv2.imshow('Creating DatasetImages', img)
        cv2.waitKey(10)

cv2.destroyAllWindows()
