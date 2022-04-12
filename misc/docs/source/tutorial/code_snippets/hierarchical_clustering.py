import io
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance

from obspy.imaging.cm import obspy_sequential


url = "https://examples.obspy.org/dissimilarities.npz"
with io.BytesIO(urlopen(url).read()) as fh, np.load(fh) as data:
    dissimilarity = data['dissimilarity']

plt.subplot(121)
plt.imshow(1 - dissimilarity, interpolation='nearest', cmap=obspy_sequential)

dissimilarity = distance.squareform(dissimilarity)
threshold = 0.3
linkage = hierarchy.linkage(dissimilarity, method="single")
clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

# A little nicer set of colors.
cmap = plt.get_cmap('Paired', lut=6)
colors = ['#%02x%02x%02x' % tuple(int(col * 255) for col in cmap(i)[:3])
          for i in range(6)]
try:
    hierarchy.set_link_color_palette(colors[1:])
except AttributeError:
    # Old version of SciPy
    pass

plt.subplot(122)
try:
    hierarchy.dendrogram(linkage, color_threshold=0.3,
                         above_threshold_color=cmap(0))
except TypeError:
    # Old version of SciPy
    hierarchy.dendrogram(linkage, color_threshold=0.3)
plt.xlabel("Event number")
plt.ylabel("Dissimilarity")
plt.show()
