import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')

# Leer etiquetas ("0 A", "1 B", ...) -> ["A", "B", ...]
with open('Model/labels.txt') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f if line.strip()]

imgSize = 300
offset = 20

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    imgClean = img.copy()  # copia sin dibujos, para clasificar
    hands, img = detector.findHands(img)

    if hands:
        x, y, w, h = hands[0]['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = imgClean[max(0, y - offset):y + h + offset,
                           max(0, x - offset):x + w + offset]

        if imgCrop.size != 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index]

            cv2.rectangle(imgOutput, (x - offset, y - offset - 60),
                          (x - offset + 160, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, f'{labels[index]} {confidence * 100:.0f}%',
                        (x - offset + 5, y - offset - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 3)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()