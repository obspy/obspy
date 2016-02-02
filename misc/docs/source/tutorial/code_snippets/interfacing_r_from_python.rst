=========================
Interfacing R from Python
=========================

The rpy2_ package allows to interface R_ from Python. The following example
shows how to convert data (:class:`numpy.ndarray`) to an R matrix and execute
the R command ``summary`` on it. 

.. doctest::

    >>> from obspy.core import read
    >>> import rpy2.robjects as RO
    >>> import rpy2.robjects.numpy2ri
    >>> r = RO.r
    >>> st = read("test/BW.BGLD..EHE.D.2008.001")
    >>> M = RO.RMatrix(st[0].data)
    >>> print(r.summary(M))  # doctest: +NORMALIZE_WHITESPACE
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
    -1056.0  -409.0  -393.0  -393.7  -378.0   233.0


.. _rpy2: http://rpy.sourceforge.net/rpy2.html
.. _R: https://www.r-project.org/
