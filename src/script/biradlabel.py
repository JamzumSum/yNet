import os
import re
import yaml

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


def malignant_label():
    M = os.listdir('./data/BIRADs/raw/malignant')
    B = os.listdir('./data/BIRADs/raw/benign')
    C = os.listdir('./data/BIRADs/raw/benign/BIRAD-2')


    mlabel = {i[:-4]: 1 for i in M if i.endswith('.jpg')}
    mlabel.update({i[:-4]: 0 for i in B if i.endswith('.jpg')})
    mlabel.update({i[:-4]: 0 for i in C})
    return mlabel

with open('./data/BIRADs/crafted/labels.yml', 'w') as f:
    yaml.safe_dump_all([
        malignant_label(), 
        birad_label()
    ], f)