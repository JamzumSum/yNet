import os
from shutil import copyfile, move, copytree, rmtree

import cv2 as cv
import numpy as np
import yaml

clearDir = lambda d: rmtree(d, ignore_errors=True) or os.makedirs(d, exist_ok=True)

def check_annotation(path):
    DD = os.path.join(path, 'B')
    pics = os.listdir(DD)
    with open(os.path.join(path, 'labels.yml')) as f:
        _, ans = yaml.safe_load_all(f)
        for i in pics:
            if i[:-4] in ans: continue
            DDP = os.path.join(DD, i)
            ODP = os.path.join(path, 'XPB', i)
            move(DDP, ODP)
            print(DDP, '->', ODP)

def cropVoid(path, output):
    rmtree(os.path.join(output, 'case'), ignore_errors=True)
    for i in ['B', 'XPB']: clearDir(os.path.join(output, i))

    for dataset in ['benign', 'malignant']:
        DD = os.path.join(path, dataset)
        for i in os.listdir(DD):
            if not i.endswith('.jpg'): continue
            
            DDP = os.path.join(DD, i)
            ODP = os.path.join(output, 'B', i)

            if i.startswith('XPB'):
                # copyfile(DDP, os.path.join(output, 'XPB', i))
                print(DDP, 'skipped.')
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

D = './data/BIRADs/raw/'
OD = './data/BIRADs/crafted'
cropVoid(D, OD)
check_annotation(OD)
copytree(os.path.join(D, 'benign', 'BIRAD-2'), os.path.join(OD, 'case'))