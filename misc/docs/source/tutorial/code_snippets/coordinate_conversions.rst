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
    >>> print(x)
    [4443947.179347951, 4446185.667319892]
    >>> print(y)
    [5506428.401023342, 5336354.054996853]

.. _`pyproj`: https://pypi.python.org/pypi/pyproj
.. _`EPSG`: https://www.epsg-registry.org/
