import os

import cv2 as cv

d = os.path.dirname(__file__)
for i in os.listdir(d):
    if not i.endswith('.pgm'): continue
    img = cv.imread(os.path.join(d, i), -1)
    if i is None:
        print('Read Failed: ' + i)
        continue
    cv.imshow('view animation', img)
    cv.waitKey(500)
