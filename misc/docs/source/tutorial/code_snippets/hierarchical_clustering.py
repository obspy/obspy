import urllib

import numpy as np
import matplotlib.pyplot as plt

import hcluster

url = "http://examples.obspy.org/dissimilarities.npz"
with np.load(urllib.urlopen(url)) as data:
    dissimilarity = data['dissimilarity']

plt.subplot(121)
plt.imshow(1 - dissimilarity, interpolation="nearest")

dissimilarity = hcluster.squareform(dissimilarity)
threshold = 0.3
linkage = hcluster.linkage(dissimilarity, method="single")
clusters = hcluster.fcluster(linkage, 0.3, criterion="distance")

plt.subplot(122)
hcluster.dendrogram(linkage, color_threshold=0.3)
plt.xlabel("Event number")
plt.ylabel("Dissimilarity")
plt.show()
