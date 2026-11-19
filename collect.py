import cv2
import numpy as np
import math
import os
import string
from cvzone.HandTrackingModule import HandDetector

# Controles:
#   Tecla A-Z  -> elegir la letra a grabar
#   ESPACIO    -> guardar imagen
#   ESC        -> salir

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

imgSize = 300
offset = 20
base_folder = 'Data'

# Crear una carpeta por letra y contar imágenes existentes
counts = {}
for letter in string.ascii_uppercase:
    path = os.path.join(base_folder, letter)
    os.makedirs(path, exist_ok=True)
    counts[letter] = len(os.listdir(path))

current = 'A'

while True:
    success, img = cap.read()
    if not success:
        break

    imgClean = img.copy()  # copia sin dibujos, para guardar
    hands, img = detector.findHands(img)
    imgWhite = None

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

            cv2.imshow('ImageWhite', imgWhite)
        else:
            imgWhite = None

    cv2.putText(img, f'Letra: {current}  Imagenes: {counts[current]}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif ord('a') <= key <= ord('z'):
        current = chr(key).upper()
    elif key == ord(' ') and imgWhite is not None:
        counts[current] += 1
        cv2.imwrite(f'{base_folder}/{current}/{current}_{counts[current]}.jpg', imgWhite)
        print(f'Guardada {current}: {counts[current]}')

cap.release()
cv2.destroyAllWindows()