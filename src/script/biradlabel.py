import os
import re
import yaml

def birad_label():
    '''
    B83 IGNORED
    B99 PASSED
    B101 PASSED
    B205 PASSED
    B224 PASSED
    '''
    csv = {}
    with open('./data/BIRADs/raw/BIRADs.csv', encoding='utf8') as f:
        f.readline()
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

    with open('./data/BIRADs/crafted/BIRADs.yml', 'w') as f:
        yaml.safe_dump(csv, f)


def malignant_label():
    M = os.listdir('./data/BIRADs/raw/malignant')
    B = os.listdir('./data/BIRADs/raw/benign')


    d = {i[:-4]: 1 for i in M if i.endswith('.jpg')}
    d.update({i[:-4]: 0 for i in B if i.endswith('.jpg')})
    with open('./data/BIRADs/crafted/malignant.yml', 'w') as f:
        yaml.safe_dump(d, f)

malignant_label()