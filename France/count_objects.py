import cv2
import numpy as np
from pathlib import Path
import math

ROOT = Path(r"e:\hackaton\images")
FOLDERS = ["apples","soda_cans","mugs","cars","forks","pears"]

def count_objects_in_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur and enhance edges
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # use adaptive threshold
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,5)
    # morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # dilate to join parts of objects
    dil = cv2.dilate(opening, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    h,w = gray.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200: # ignore tiny
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        # ignore very large full-image components
        if ww > 0.9*w and hh > 0.9*h:
            continue
        # heuristics: aspect ratio constraints
        ar = ww / float(hh)
        # accept wide or tall shapes
        if area > 500:
            count += 1
    return count


def a1z26(num):
    # map 1->A ... 26->Z, if outside, wrap modulo
    if num <= 0:
        return '?'
    num0 = ((num - 1) % 26) + 1
    return chr(ord('A') + num0 - 1)

if __name__ == '__main__':
    totals = {}
    for folder in FOLDERS:
        p = ROOT / folder
        if not p.exists():
            print(f"Folder not found: {p}")
            continue
        total = 0
        files = sorted([x for x in p.iterdir() if x.suffix.lower() in ['.png','.jpg','.jpeg']])
        for f in files:
            c = count_objects_in_image(f)
            total += c
            # debug
            print(f"{folder}/{f.name}: {c}")
        totals[folder] = total

    print('\nTotals:')
    for k in FOLDERS:
        print(k, totals.get(k,0), a1z26(totals.get(k,0)))
    # construct word by ordering folders as in FOLDERS
    word = ''.join(a1z26(totals.get(k,0)) for k in FOLDERS)
    print('\nCandidate word:', word)
    # print flag
    print('\nFLAG: SIGMOID_{%s}' % word)
