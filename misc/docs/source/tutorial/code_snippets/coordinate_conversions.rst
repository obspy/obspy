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


Another common usage is to convert location information in latitude and
longitude to `Universal Transverse Mercator coordinate system (UTM)`_. This is
especially useful for large dense arrays in a small area. Such conversion can
be easily done using `utm`_ package. Below is its typical usages:

.. doctest::

    >>> import utm
    >>> utm.from_latlon(51.2, 7.5)
    (395201.3103811303, 5673135.241182375, 32, 'U')
    >>> utm.to_latlon(340000, 5710000, 32, 'U')
    (51.51852098408468, 6.693872395145327)


.. _`Universal Transverse Mercator coordinate system (UTM)`: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
.. _`utm`: https://pypi.python.org/pypi/utm