import argparse
import os
import re

import yaml

IMAGE_EXT = ['.jpg', '.bmp', '.png']


def birad_label():
    with open('./data/BIRADs/raw/BIRADs.csv', encoding='utf8') as f:
        f.readline()
        csv = {}
        for i in f:
            k, v = i.strip().split(',')
            if v.count('类') > 1:
                print(k, v, 'PASSED')
                continue
            v = re.findall(r'(\d[abc]?)类?$', v)
            if len(v) > 1:
                print(k, v, 'PASSED')
                continue
            elif v:
                csv[k] = v[0]
            else:
                print(k, 'IGNORED')
    C = os.listdir('./data/BIRADs/raw/benign/BIRAD-2')
    csv.update({os.path.splitext(i)[0]: '2' for i in C})
    return csv


def malignant_label(folder_label: dict):
    mlabel = {}
    for folder, label in folder_label.items():
        M = os.listdir(folder)
        for i in M:
            name, ext = os.path.splitext(i)
            if ext in IMAGE_EXT: mlabel[name] = label
    return mlabel


def mask_map(datasets):
    dic = {}
    for D in datasets:
        for i in os.listdir(D):
            if os.path.isdir(os.path.join(D, i)): continue
            dic[os.path.splitext(i)[0]] = []
    for D in datasets:
        D = os.path.join(D, 'mask')
        if not os.path.exists(D): continue
        for i in os.listdir(D):
            g = re.match(r'(.+)_mask(_\d)?$', os.path.splitext(i)[0])
            dic[g.group(1)].append(os.path.join(D, i))
    return dic


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('folder', type=str)
    psr.add_argument(
        '--sets',
        nargs='+',
        type=lambda s: s.split(':'),
        default=[('malignant', 1), ('benign', 0)]
    )
    psr.add_argument('--title', nargs='+', type=str)
    arg = psr.parse_args()

    outpath = './data/%s/crafted' % arg.folder
    inpath = './data/%s/raw' % arg.folder
    if not os.path.exists(inpath):
        print('%s not found. Inpath set to %s.' % (inpath, outpath))
        inpath = outpath

    dumplist = [None] * 3
    BMdic = {os.path.join(inpath, k): int(v) for k, v in arg.sets}
    if 'Ym' in arg.title:
        dumplist[0] = malignant_label(BMdic)
    if 'Yb' in arg.title:
        assert arg.folder == 'BIRADs', 'BIRADs supports ourset only.'
        dumplist[1] = birad_label()
    if 'mask' in arg.title:
        assert arg.folder == 'BUSI', 'segment mask supports BUSI only'
        dumplist[2] = mask_map(BMdic.keys())

    if not all(i or i is None for i in dumplist):
        raise RuntimeWarning('At least an item is empty.')
    with open(outpath + '/labels.yml', 'w') as f:
        yaml.safe_dump_all(dumplist, f)
