import cv2
import glob as gb
import numpy as np
import os
from main import PreProcess

images = gb.glob('training/*/*')
labels = []
training_set = []

test_img = PreProcess('leaves/BW_OLD/BW_OLD58.jpg')
similar = []

for fileName in images:
    print(".", end='')
    directory = os.path.dirname(fileName)
    name = directory.replace('training\\', '')
    I = cv2.imread(fileName)
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    h, w = Igray.shape[:2]
    similarity = cv2.matchTemplate(test_img, Igray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    location = np.where(similarity>threshold)

    for point in zip(*location[::-1]):
        print("!", end='')
        labels.append(name.lower())
        similar.append(similarity)
        print('The test image is {} similar to {}'.format(str(100*similarity[0][0]), name.lower()))
