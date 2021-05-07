import argparse
import os, re
from collections import defaultdict

import cv2 as cv
import numpy as np
import torch
import yaml
from misc.indexserial import IndexDumper

BIRAD_NAME = ['3', '4', '5']
BIRAD_MAP = {
    '3': 0,
    '4a': 1,
    '4b': 1,
    '4c': 1,
    '5': 2,
}
M_NAME = ["bengin", "malignant"]


def makecache(inpath, outpath, datasets: list, title: list, statTitle=None):
    assert datasets
    if "pid" not in title:
        title.insert(0, "pid")
    if statTitle is None:
        statTitle = []
    shapedic = defaultdict(lambda: {k: [] for k in title})
    statdic = {k: defaultdict(int) for k in statTitle}

    dumper = IndexDumper(os.path.join(outpath, "images.pts"))
    with open(os.path.join(inpath, "labels.yml")) as f:
        mdic, bdic, maskdic = yaml.safe_load_all(f)

    idxs = []
    for D in datasets:
        DD = os.path.join(inpath, D)
        pics = os.listdir(DD)
        for p in pics:
            DDP = os.path.join(DD, p)
            if os.path.isdir(DDP):
                continue
            fname, _ = os.path.splitext(p)
            img = cv.imread(DDP, 0)
            if img is None:
                raise ValueError(DDP)
            assert not np.any(np.isnan(img)), DDP
            dic = shapedic[img.shape]

            if 'Yb' in dic and bdic[fname] not in BIRAD_MAP: continue

            img = torch.from_numpy(img).unsqueeze(0)
            img = img / img.max()
            idxs.append(dumper.dump(img))

            dic["pid"].append(fname)
            if "X" in dic:
                dic["X"].append(len(idxs) - 1)
            if "Ym" in dic:
                dic["Ym"].append(mdic[fname])
            if "Yb" in dic:
                dic["Yb"].append(BIRAD_MAP[bdic[fname]])
            if "mask" in dic:
                masks = []
                for p in maskdic[fname]:
                    img = cv.imread(p, 0)
                    assert not np.any(np.isnan(img)), p
                    img = torch.from_numpy(img).unsqueeze(0)
                    masks.append(img > 0)
                masks = sum(masks).float()
                idxs.append(dumper.dump(masks))
                dic["mask"].append(len(idxs) - 1)

    for dic in shapedic.values():
        if "mask" in dic:
            assert len(dic["mask"]) == len(dic["X"])
        for t in ["Ym", "Yb"]:
            if t in dic:
                if t in statdic:
                    for p in dic[t]:
                        statdic[t][p] += 1
                dic[t] = torch.LongTensor(dic[t])
    for k, v in statdic.items():
        statdic[k] = torch.IntTensor([v[p] for p in range(max(v) + 1)])

    torch.save(
        {
            "data": dict(shapedic),
            "index": idxs,
            "title": title,
            "statistic_title": statTitle,
            "classname": {
                "Ym": M_NAME,
                "Yb": BIRAD_NAME
            },
            "distribution": statdic,
        },
        os.path.join(outpath, "meta.pt"),
    )
    return shapedic


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("folder", type=str)
    psr.add_argument("--sets", nargs="+", type=str)
    psr.add_argument("--title", nargs="+", type=str)
    psr.add_argument("--stat", nargs="+", type=str)

    arg = psr.parse_args()

    inpath = "./data/%s/crafted" % arg.folder
    outpath = "./data/%s" % arg.folder
    if arg.sets:
        datasets = arg.sets
    else:
        datasets = [
            i for i in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, i))
        ]

    dic = makecache(
        inpath, outpath, datasets, title=["X"] + arg.title, statTitle=arg.stat
    )
    for k, v in dic.items():
        print(k, len(v["X"]), "items cached")
