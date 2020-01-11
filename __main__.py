#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.detector import Detector
from src.recognizer import Recognizer
import os
import sys
import cv2

os.makedirs('test', exist_ok=True)

image = cv2.imread(sys.argv[1])

detector = Detector()
detector.load()

recognizer = Recognizer()
recognizer.load()

roi, _, _, _ = detector.process(image)

for i, img in enumerate(roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 5)
    # thresh = cv2.adaptiveThreshold(
    #     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('test/%.2d.jpg' % i, gray)

    text, _, _ = recognizer.process(gray)
    print('%.2d' % i, text)
# end for
