import os
from shutil import copyfile

import cv2 as cv
import numpy as np

D = './data/BIRADs/raw/'
OD = './data/BIRADs/crafted'

for dataset in ['benign', 'malignant']:
    DD = os.path.join(D, dataset)
    for i in os.listdir(DD):
        if not i.endswith('.jpg'): continue
        
        DDP = os.path.join(DD, i)
        ODP = os.path.join(OD, 'B', i)

        if i.startswith('XPB'):
            copyfile(DDP, os.path.join(OD, 'XPB', i))
            print(DDP, 'copied.')
            continue

        img = cv.imread(DDP, 0)

        _, tmg = cv.threshold(img, 4, 255, cv.THRESH_BINARY)
        v = np.count_nonzero(tmg, axis=0) / tmg.shape[0]

        d = v.shape[0] // 2
        lv = np.argwhere(v[:4] < 0.3)
        rv = np.argwhere(v[d:] < 0.3)
        f = t = 0
        if lv.shape[0]: f = lv.max()
        if rv.shape[0]: t = rv.min()
        if not (f + t): 
            copyfile(DDP, ODP)
            print(DDP, 'copied.')
        if t: t += d
        else: t = v.shape[0]

        rimg = img[:, f: t]
        cv.imwrite(ODP, rimg)
        print(DDP, img.shape, '->', rimg.shape)
