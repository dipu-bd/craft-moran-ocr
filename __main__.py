#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
from PIL import Image
from pytesseract import image_to_string

# from src.detector import Detector

# image = cv2.imread(sys.argv[1])

# detector = Detector()
# detector.load()
# roi, _, _, _ = detector.process(image)

# os.makedirs('test', exist_ok=True)
# for i, img in enumerate(roi):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.medianBlur(gray, 5)
#     thresh = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     cv2.imwrite('test/%.2d.jpg' % i, thresh)
    
#     text = image_to_string(Image.fromarray(img), lang='ben')
#     print('%.2d' % i, text)
# # end for

image = cv2.imread(sys.argv[1], 0)
# image = cv2.medianBlur(image, 3)
#_, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# image = cv2.adaptiveThreshold(
#         image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
pil_image = Image.fromarray(image)
print(image_to_string(pil_image))
