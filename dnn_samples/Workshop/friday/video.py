import cv2
import glob as gb
import numpy as np
import os

cam = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def mema():
    while True:
        ret, frame = cam.read()
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # frame[:,:,0] = cv2.equalizeHist(frame[:,:,0])
        # ret, Ibin = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        # Ibin = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # cv2.imshow("Camera", np.concatenate((frame, Ibin), axis=1))
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('logos/logo1.jpg', frame)
            break

while True:
    ret, frame = cam.read()    
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('logos/logo1.jpg', frame)
        break

cam.release()
cv2.destroyAllWindows()