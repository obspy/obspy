=======================
Hierarchical Clustering
=======================

An implementation of hierarchical clustering is provided in the package
`hcluster`_. Among other things, it allows to build clusters from similarity
matrices and make dendrogram plots. The following example shows how to do this
for an already computed similarity matrix. The similarity data are computed
from events in an area with induced seismicity (using the cross-correlation
routines in :mod:`obspy.signal`) and can be fetched from our
`examples webserver`_:

First, we import the necessary modules and load the data stored on our
webserver:

.. doctest::

    >>> import pickle, urllib
    >>> import matplotlib.pyplot as plt
    >>> import hcluster
    >>> 
    >>> url = "http://examples.obspy.org/dissimilarities.pkl"
    >>> dissimilarity = pickle.load(urllib.urlopen(url))

Now, we can start building up the plots. First, we plot the dissimilarity
matrix:

.. doctest::

    >>> plt.subplot(121)
    >>> plt.imshow(1 - dissimilarity, interpolation="nearest")

After that, we use `hcluster`_ to build up and plot the dendrogram into the
right-hand subplot:

.. doctest::

    >>> dissimilarity = hcluster.squareform(dissimilarity)
    >>> threshold = 0.3
    >>> linkage = hcluster.linkage(dissimilarity, method="single")
    >>> clusters = hcluster.fcluster(linkage, 0.3, criterion="distance")
    >>> 
    >>> plt.subplot(122)
    >>> hcluster.dendrogram(linkage, color_threshold=0.3)
    >>> plt.xlabel("Event number")
    >>> plt.ylabel("Dissimilarity")
    >>> plt.show()

.. plot:: tutorial/code_snippets/hierarchical_clustering.py


.. _`hcluster`: http://pypi.python.org/pypi/hcluster
.. _`examples webserver`: http://examples.obspy.org
