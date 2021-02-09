import os
from collections import defaultdict

import cv2 as cv
import numpy as np
import torch
import yaml
from utils.indexserial import IndexDumper

BIRAD_NAME = ['2', '3', '4a', '4b', '4c', '5']
M_NAME = ['bengin', 'malignant']

def FSCache(inpath, outpath, datasets, title, statTitle=[]):
    shapedic = defaultdict(lambda: {k: [] for k in title})
    statdic = {k: defaultdict(int) for k in statTitle}

    bans = mans = None
    with open(os.path.join(inpath, 'labels.yml')) as f: mans, bans = yaml.safe_load_all(f)
    dumper = IndexDumper(os.path.join(outpath, 'ourset.imgs'))

    idxs = []
    for D in datasets:
        DD = os.path.join(inpath, D)
        pics = os.listdir(DD)
        for i in pics:
            DDP = os.path.join(DD, i)
            img = cv.imread(DDP, 0)
            dic = shapedic[img.shape]

            img = torch.from_numpy(img).unsqueeze(0)
            img = img / img.max()
            idxs.append(dumper.dump(img))
            Yb = BIRAD_NAME.index(bans[i[:-4]])

            append_data = lambda t, d: (t in dic) and dic[t].append(d)
            append_data('X', len(idxs) - 1)
            append_data('Ym', mans[i[:-4]])
            append_data('Yb', Yb)

    for dic in shapedic.values():
        for t in ['Ym', 'Yb']:
            if t in dic:
                if t in statdic:
                    for i in dic[t]: statdic[t][i] += 1
                dic[t] = torch.LongTensor(dic[t])
    for k, v in statdic.items():
        statdic[k] = torch.LongTensor([v[i] for i in range(max(v.keys()) + 1)])
                
    torch.save(
        {
            'data': dict(shapedic), 
            'index': idxs,
            'title': title,
            'statistic_title': statTitle,
            'classname': {
                'Ym': M_NAME, 
                'Yb': BIRAD_NAME
            }, 
            'distribution': statdic
        }, 
        os.path.join(outpath, 'ourset.pt')
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
    dic = FSCache(
        inpath='./data/BIRADs/crafted', 
        outpath='./data/BIRADs', 
        datasets=['B', 'case'], 
        title=['X', 'Ym', 'Yb'], 
        statTitle=['Ym', 'Yb']
    )
    for k, v in dic.items():
        print(k, len(v['X']), 'items cached')
