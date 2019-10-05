from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
import math

min_confidence = 0.9
padding = 10
east = 'frozen_east_text_detection.pb'

def decode_predictions(scores, geometry):
    numRows, numCols = scores.shape[:2]
    rects = []
    confidences = []

    for y in range(0, numRows):
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX, offsetY = (x*4.0, y*4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos*xData1[x]) + (sin*xData2[x]))
            endY = int(offsetY + (sin*xData1[x]) + (cos*xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences

def process_image(image):
    orig = image.copy()
    origH, origW = image.shape[:2]

    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115,1)

    image[:,:,0] = th
    image[:,:,1] = th
    image[:,:,2] = th

    width = math.floor(origW/32)*32
    height = math.floor(origH/32)*32

    newW, newH = (width, height)
    rW = origW/float(newW)
    rH = origH/float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    # print("Reading Text per Frame")
    net = cv2.dnn.readNet(east)

    blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), (123.68,116.78,103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(layerNames)
    rects, confidences = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    results = []

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX*rW)
        startY = int(startY*rH)
        endX = int(endX*rW)
        endY = int(endY*rH)

        dX = int((endX - startX)*padding)
        dY = int((endY - startY)*padding)
        
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = max(origW, endX + 2*dX)
        endY = max(origH, endY + 2*dY)

        roi = orig[startY:endY, startX:endX]
        text = pytesseract.image_to_string(roi, config=("-l end --oem 3 --psm 12"))

        results.append((startX, startY, endX, endY), text)

    results = sorted(results, key=lambda r:r[0][1])
    for ((startX, startY, endX, endY), text) in results:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

    return text

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    # frame = cv2.imread('img.jpg')

    try:
        text = process_image(frame)
        print(text)
        done = True
        # cv2.putText(frame, text, (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    except:
        print(".", end='')
        pass

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow("Camera", frame)

    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()