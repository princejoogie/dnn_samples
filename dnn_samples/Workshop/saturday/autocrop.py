import cv2
import glob as gb
import numpy as np

images = gb.glob('fresh/*.*')
faceCascade = cv2.CascadeClassifier('GillClassifier.xml')
n = 1

for nFile in images:
    frame = cv2.imread(nFile)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (w+x, y+h), (0,255,0), 2)
        gill = cv2.resize(frame[y:y+h, x:x+w], (300, 300))
        fileName = "notfreshCropped/not_fresh{}{}".format(str(n), ".jpg")
        print(fileName)
        cv2.imwrite(fileName, gill)
        n += 1
