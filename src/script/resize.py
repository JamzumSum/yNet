import cv2 as cv
import numpy as np
import os

D = './data/BIRADs/crafted'

def resize(dataset):
    DD = os.path.join(D, dataset)
    pics = os.listdir(DD)
    for i in pics:
        DDP = os.path.join(DD, i)
        img = cv.imread(DDP)
        f = 512 / min(img.shape[:2])
        rimg = cv.resize(img, None, fx=f, fy=f)
        assert 512 == min(rimg.shape[:2])

        r, c, _ = rimg.shape
        r = (r - 512) // 2
        c = (c - 512) // 2
        rimg = rimg[r: r + 512, c: c + 512] # [512, 512, 3]
        cv.imwrite(DDP, rimg)
        print(DDP, img.shape, '->', rimg.shape)

if __name__ == "__main__":
    for i in ['B', 'XPB', 'case']:
        resize(i)