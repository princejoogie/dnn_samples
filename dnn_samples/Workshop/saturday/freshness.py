
from sklearn.svm import SVC
import numpy as np
import pickle
import cv2
import time
import pywt

clf = SVC()

with open('freshness_model.pickle', 'rb') as f:
    features, labels = pickle.load(f)

clf.fit(features, labels)

def getMeanRGB(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    average_color = cA.ravel()
    return average_color

cam = cv2.VideoCapture(0)
classes = ["fresh", "not_fresh"]
faceCascade = cv2.CascadeClassifier('cascades/gill_classifier.xml')

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (w+x, y+h), (0,255,0), 2)
        gill = cv2.resize(frame[y:y+h, x:x+w], (300, 300))
        feature = getMeanRGB(gill)
        idx = clf.predict([feature, feature])
        cv2.putText(frame, classes[idx[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()