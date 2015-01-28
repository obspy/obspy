import io
import urllib

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial import distance

url = "http://examples.obspy.org/dissimilarities.npz"
with io.BytesIO(urllib.urlopen(url).read()) as fh:
    with np.load(fh) as data:
        dissimilarity = data['dissimilarity']

plt.subplot(121)
plt.imshow(1 - dissimilarity, interpolation="nearest")

dissimilarity = distance.squareform(dissimilarity)
threshold = 0.3
linkage = hierarchy.linkage(dissimilarity, method="single")
clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

plt.subplot(122)
hierarchy.dendrogram(linkage, color_threshold=0.3)
plt.xlabel("Event number")
plt.ylabel("Dissimilarity")
plt.show()
