import cv2 as cv
import numpy as np
import os
import yaml
import argparse


def CentralCrop(inpath, dataset):
    DD = os.path.join(inpath, dataset)
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
        datasets = [i for i in os.listdir(inpath) if os.path.isdir(i)]

    for i in datasets:
        CentralCrop(inpath, i)
    # resizeWithCluster('./data/BIRADs/crafted/boxcluster.yml')