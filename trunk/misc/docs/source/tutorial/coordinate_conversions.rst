======================
Coordinate Conversions
======================

Coordinate conversions can be done conveniently using `pyproj`_. After looking
up the `EPSG`_ codes of source and target coordinate system, the conversion can
be done in just a few lines of code. The following example converts the station
coordinates of two German stations to the regionally used Gauß-Krüger system:

.. doctest::

    >>> import pyproj
    >>> lat = [49.6919, 48.1629]
    >>> lon = [11.2217, 11.2752]
    >>> proj_wgs84 = pyproj.Proj(init="epsg:4326")
    >>> proj_gk4 = pyproj.Proj(init="epsg:31468")
    >>> x, y = pyproj.transform(proj_wgs84, proj_gk4, lon, lat)
    >>> print x, y
    [4443947.179, 4446185.667] [5506428.401, 5336354.055]

.. _`pyproj`: http://pypi.python.org/pypi/pyproj
.. _`EPSG`: http://www.epsg-registry.org/