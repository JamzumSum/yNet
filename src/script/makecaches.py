import os

import cv2 as cv
import numpy as np
import torch
import yaml
from collections import defaultdict

BIRAD_NAME = ['2', '3', '4a', '4b', '4c', '5']

def FSCache(inpath, outpath, datasets):
    shapedic = defaultdict(list)
    bans = mans = None
    with open(os.path.join(inpath, 'labels.yml')) as f: mans, bans = yaml.safe_load_all(f)

    for D in datasets:
        DD = os.path.join(inpath, D)
        pics = os.listdir(DD)
        for i in pics:
            DDP = os.path.join(DD, i)
            img = cv.imread(DDP, 0)
            img = torch.from_numpy(cv.equalizeHist(img))
            img = img / img.max()
            birad = BIRAD_NAME.index('2' if D == 'case' else bans[i[:-4]])

            shapedic[img.shape].append((
                img, mans[i[:-4]], birad
            ))

    for k, v in shapedic.items():
        s = sorted(v, key=lambda t: t[2])
        X, Ym, Yb = ([t[i] for t in s] for i in range(3))
        X = torch.stack(X).unsqueeze_(1)
        Ym = torch.LongTensor(Ym)
        Yb = torch.LongTensor(Yb)
        shapedic[k] = (X, Ym, Yb)

    torch.save(
        {
            'data': shapedic, 
            'classname': [
                ['bengin', 'malignant'], 
                BIRAD_NAME
            ]
        }, 
        os.path.join(outpath, 'annotated.pt')
    )
    return shapedic

def WSCache(inpath, outpath):
    DD = os.path.join(inpath, 'XPB')
    mans = None
    with open(os.path.join(inpath, 'labels.yml')) as f: mans, _ = yaml.safe_load_all(f)

    shapedic = defaultdict(list)
    for i in os.listdir(DD):
        img = cv.imread(os.path.join(DD, i), 0)
        img = torch.from_numpy(cv.equalizeHist(img))
        shapedic[img.shape].append((
            img, mans[i[:-4]]
        ))

    for k, v in shapedic.items():
        s = sorted(v, key=lambda t: t[1])
        X, Ym = ([t[i] for t in s] for i in range(2))
        X = torch.stack(X).unsqueeze_(1)
        Ym = torch.LongTensor(Ym)
        shapedic[k] = (X, Ym)
    
    torch.save(
        {
            'data': shapedic, 
            'classname': [
                ['bengin', 'malignant'], 
            ]
        }, 
        os.path.join(outpath, 'unannotated.pt')
    )
    return shapedic

if __name__ == "__main__":
    dic = FSCache('./data/BIRADs/crafted', './data/BIRADs', ['B', 'case'])
    for k, v in dic.items():
        print(k, len(v[0]), 'items cached')