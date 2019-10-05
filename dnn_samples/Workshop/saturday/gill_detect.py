import cv2
import glob as gb
import os
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle
import pywt

images = gb.glob('training/*/*.*')

def getMeanRGB(filename):
    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,50,50), (10,255,255))
    mask2 = cv2.inRange(hsv, (170,50,50), (180,255,255))
    mask = mask1 + mask2
    img = cv2.bitwise_and(img, img, mask=mask)
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    average_color = cA.ravel()
    directory = os.path.dirname(filename)
    classify = directory.replace('training\\','')
    return average_color, classify

labels = []
features = []

for file in images:
    print(file)
    bgr, cfy = getMeanRGB(file)
    labels.append(cfy)
    features.append(bgr)

le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

clf = SVC()
clf.fit(features, labels)

with open('freshness_model.pickle', 'wb') as f:
    pickle.dump([features, labels], f)