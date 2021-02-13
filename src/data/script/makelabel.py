import argparse
import os
import re

import yaml

IMAGE_EXT = ['.jpg', '.bmp']

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
            elif v: csv[k] = v[0]
            else:
                print(k, 'IGNORED')
    C = os.listdir('./data/BIRADs/raw/benign/BIRAD-2')
    csv.update({i[:-4]: '2' for i in C})
    return csv


def malignant_label(folder_label: dict):
    mlabel = {}
    for folder, label in folder_label.items():
        M = os.listdir(folder)
        for i in M:
            name, ext = os.path.splitext(i)
            if ext in IMAGE_EXT: mlabel[name] = label
    return mlabel

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('folder', type=str)
    psr.add_argument('--sets', nargs='+', type=lambda s: s.split(':'), default=[('malignant', 1), ('benign', 0)])
    psr.add_argument('--title', nargs='+', type=str)
    arg = psr.parse_args()

    outpath = './data/%s/crafted' % arg.folder
    inpath = './data/%s/raw' % arg.folder
    if not os.path.exists(inpath): 
        print('%s not found. Inpath set to %s.' % (inpath, outpath))
        inpath = outpath

    dumplist = [None] * 2
    if 'Ym' in arg.title:
        dumplist[0] = malignant_label({os.path.join(inpath, k): v for k, v in arg.sets})
    if 'Yb' in arg.title:
        assert arg.folder == 'BIRADs', 'BIRADs supports ourset only.'
        dumplist[1] = birad_label()

    with open(outpath + '/labels.yml', 'w') as f:
        yaml.safe_dump_all(dumplist, f)
