import cv2
import glob as gb
import numpy as np
import os
import traceback

def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des, image.shape[:2][::-1]

def detectFeatures(image, train_features):
    train_kps, train_desc, shape = train_features
    kps, desc, _ = getFeatures(image)

    if not kps:
        return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_desc, desc, k=2)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])

        if len(good) < 0.1*len(train_kps):
            return None
        
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if m is not None:
            scene_points = cv2.perspectiveTransform(np.float32(
                [(0,0), (0, shape[0] - 1), 
                (shape[1] - 1, shape[0] - 1),
                (shape[1] - 1, 0)]
            ).reshape(-1, 1, 2), m)

            rect = cv2.minAreaRect(scene_points)

            if rect[1][1] > 0 and 0.8 < rect[1][0] / rect[1][1] < 1.2:
                return rect
    except:
        pass
    return None

cam = cv2.VideoCapture(0)

img = cv2.imread('logos/logo1.jpg')
train_features = getFeatures(img)

while True:
    ret, frame = cam.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    region = detectFeatures(frame, train_features)

    if region is not None:
        box = cv2.boxPoints(region)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0,255,0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows() 