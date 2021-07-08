#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from src.detector import Detector
from src.recognizer import Recognizer
import os
import sys
import cv2
from shutil import rmtree

OUTPUT_DIR = 'output'

rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test(image_file):
    image = cv2.imread(image_file)

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

        text, _, _ = recognizer.process(gray)

        out_file = '%s/%.2d_%s.jpg' % (OUTPUT_DIR, i, text)
        cv2.imwrite(out_file, gray)

    print('Saved OCR result to "%s" folder' % OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', nargs='+')
    args = parser.parse_args()
    for filename in args.image_files:
        test(filename)


if __name__ == '__main__':
    main()
