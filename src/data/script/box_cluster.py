import os
from collections import OrderedDict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml

dic = OrderedDict()
D = './data/BIRADs/crafted/'

def cluster(datasets, name, K):
    for dataset in datasets:
        DD = os.path.join(D, dataset)
        for i in os.listdir(DD):
            DDP = os.path.join(DD, i)
            dic[DDP] = cv.imread(DDP, 0).shape

    raw = np.array(list(dic.values()))
    data = raw.astype(np.float)
    data = np.concatenate([data[:, 0:1] / data[:, 1:2], data], axis=1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    model = KMeans(n_clusters=K, init='k-means++')
    y = model.fit_predict(data)

    center = scaler.inverse_transform(model.cluster_centers_)
    scales = center[:, 0]
    center = center[:, 1:]
    stat = np.bincount(y)
    for i, s in enumerate(stat):
        if s < 8: 
            to = np.argsort(np.abs(scales - scales[i]))[1]
            y[y == i] = to

    plt.scatter(raw[:,0], raw[:,1], c=y)
    return {k: int(v) for k, v in zip(dic.keys(), y)}, center.tolist()

if __name__ == "__main__":
    dic, center = cluster(['B', 'case'], 'raw', 6)
    round16 = lambda x: int(16 * round(x / 16))
    center = [[round16(i) for i in l] for l in center]
    with open(os.path.join(D, 'boxcluster.yml'), 'w') as f:
        yaml.safe_dump_all([center, dic], f)
    plt.show()