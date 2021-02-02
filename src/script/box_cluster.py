import os
from collections import OrderedDict

import cv2 as cv
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans

cm = matplotlib.colors.ListedColormap(list('rgbm'))

dic = OrderedDict()
D = './data/BIRADs/raw/'

for dataset in ['benign', 'malignant']:
    DD = os.path.join(D, dataset)
    for i in os.listdir(DD):
        if not i.endswith('.jpg'): continue
        DDP = os.path.join(DD, i)
        dic[DDP] = cv.imread(DDP, 0).shape

DD = os.path.join(D, 'benign', 'BIRAD-2')
for i in os.listdir(DD):
    DDP = os.path.join(DD, i)
    dic[DDP] = cv.imread(DDP, 0).shape

data = np.array(list(dic.values()))

model=KMeans(n_clusters=3,init='k-means++')
y_pre=model.fit_predict(data)

print(model.cluster_centers_)

plt.scatter(data[:,0],data[:,1],c=y_pre,cmap=cm)
plt.title('shapes cluster')
plt.grid()
plt.show()
