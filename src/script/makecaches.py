import os

import cv2 as cv
import numpy as np
import torch
import yaml

D = './data/BIRADs/crafted'
BIRAD_NAME = ['2', '3', '4a', '4b', '4c', '5']

def check_annotation():
    DD = os.path.join(D, 'B')
    pics = os.listdir(DD)
    with open(os.path.join(D, 'BIRADs.yml')) as f:
        ans = yaml.safe_load(f)
        for i in pics:
            if i[:-4] not in ans:
                print(os.path.join(DD, i))

def FSCache():
    imgs = []; isM = []; BIRADs = []
    bans = mans = None
    with open(os.path.join(D, 'BIRADs.yml')) as f: bans = yaml.safe_load(f)
    with open(os.path.join(D, 'malignant.yml')) as f: mans = yaml.safe_load(f)

    for dataset in ['B', 'case']:
        DD = os.path.join(D, dataset)
        pics = os.listdir(DD)
        for i in pics:
            DDP = os.path.join(DD, i)
            img = cv.imread(DDP)
            birad = BIRAD_NAME.index('2' if dataset == 'case' else bans[i[:-4]])

            imgs.append(img)
            isM.append(mans[i[:-4]] if dataset == 'B' else 0)
            BIRADs.append(birad)

    imgs = torch.from_numpy(np.array(imgs))         # [N, H, W, C]
    isM = torch.from_numpy(np.array(isM))
    BIRADs = torch.from_numpy(np.array(BIRADs))
    
    imgs = imgs.permute(0, 3, 1, 2)     # [N, C, H, W]

    assert imgs.shape[1] == 3
    torch.save(
        {
            'X': imgs / 255, 
            'Ym': isM,
            'Ybirad': BIRADs, 
            'cls_name': BIRAD_NAME
        }, 
        os.path.join(D, '../annotated.pt')
    )
    print('%d items cached.' % imgs.shape[0])

def WSCache():
    DD = os.path.join(D, 'XPB')
    mans = None
    with open(os.path.join(D, 'malignant.yml')) as f: mans = yaml.safe_load(f)

    X = []; Ym = []
    for i in os.listdir(DD):
        X.append(cv.imread(os.path.join(DD, i)))
        Ym.append(mans[i[:-4]])

    imgs = torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)    # [N, C, H, W]
    isM = torch.from_numpy(np.array(Ym))
    
    torch.save(
        {
            'X': imgs / 255, 
            'Ym': isM
        }, 
        os.path.join(D, '../unannotated.pt')
    )
    print('%d items cached.' % imgs.shape[0])

if __name__ == "__main__":
    # check_annotation()
    FSCache()
    WSCache()