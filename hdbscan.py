import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import HDBSCAN

dstorm_locs = pd.read_hdf("Y:\\Ross\\dSTORM\\ross dstorm -2.sld - cell3 - 3_locs_ROI.hdf5", key='locs')
dstorm_locs_XY = dstorm_locs.iloc[:, [1, 2]]
X = dstorm_locs_XY.to_numpy()

## show scatterplot of raw data
plt.figure(1)
scatter = plt.scatter(X[:, 0], X[:, 1], marker='.', s =0.9)
ax = scatter.axes
ax.invert_yaxis()


hdb = HDBSCAN(min_cluster_size=20)
hdbscan_results = hdb.fit(X)
print(hdb.labels_)
