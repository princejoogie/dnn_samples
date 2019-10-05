import cv2
import numpy as np

cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('GillClassifier.xml')

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (w+x, y+h), (0,255,0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("logo.jpg", frame)
        break

cam.release()
cv2.destroyAllWindows()
