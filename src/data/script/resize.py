import argparse
import os
from shutil import copytree

import cv2 as cv
import numpy as np
import yaml


def CentralCrop(inpath, dataset, maskdic=None):
    DD = os.path.join(inpath, dataset)
    pics = os.listdir(DD)
    for i in pics:
        pgroup = [os.path.join(DD, i)]
        if os.path.isdir(*pgroup): continue
        if maskdic: pgroup.extend(maskdic.get(os.path.splitext(i)[0], [])) 
        for DDP in pgroup:
            img = cv.imread(DDP, 0)
            
            f = 512 / min(img.shape[:2])
            rimg = cv.resize(img, None, fx=f, fy=f)
            assert 512 == min(rimg.shape[:2])

            r, c = rimg.shape
            r = (r - 512) // 2
            c = (c - 512) // 2
            rimg = rimg[r: r + 512, c: c + 512] # [512, 512]
            cv.imwrite(DDP, rimg)
            print(DDP, img.shape, '->', rimg.shape)

def resizeWithCluster(path):
    with open(path) as f:
        boxes, dic = yaml.safe_load_all(f)
        for k, v in dic.items():
            img = cv.imread(k, 0)
            rimg = cv.resize(img, tuple(boxes[v][::-1]))
            cv.imwrite(k, rimg)
            print(k, img.shape, '->', rimg.shape)

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('folder', type=str)
    psr.add_argument('--sets', nargs='+', type=str)
    arg = psr.parse_args()

    inpath = './data/%s/crafted' % arg.folder
    if arg.sets:
        datasets = arg.sets
    else:
        datasets = [i for i in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, i))]
    
    with open(os.path.join(inpath, 'labels.yml')) as f: 
        _, _, maskdic = yaml.safe_load_all(f)

    for i in datasets:
        CentralCrop(inpath, i, maskdic)
    # resizeWithCluster('./data/BIRADs/crafted/boxcluster.yml')
