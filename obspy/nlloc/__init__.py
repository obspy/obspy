#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.nllox - NonLinLoc file format support for ObsPy
=====================================================

This module provides read/write support for some NonLinLoc file
formats.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Example
-------

First, if NonLinLoc location run was performed in some custom coordinate system
(as opposed to WGS84 with depth in meters down), we need to set up a coordinate
conversion function to convert from the NonLinLoc location coordinates `x`,
`y`, `z` to longitude, latitude and depth in kilometers.
In the example, the location run was done in Gauß-Krüger zone 4 (EPSG:31468,
but in kilometers for coordinates) and depth in kilometers. So we have to
convert `x` and `y` (of NonLinLoc location run) to meters (as defined in
EPSG:31468) and then we convert to WGS84 (EPSG:4326). The `z` coordinate is
already in kilometers downwards and can be left as is. For the conversion we
use `pyproj`.

>>> import pyproj  # doctest: +SKIP
>>> proj_wgs84 = pyproj.Proj(init="epsg:4326")  # doctest: +SKIP
>>> proj_gk4 = pyproj.Proj(init="epsg:31468")  # doctest: +SKIP
>>> def my_conversion(x, y, z):
...     x *= 1e3
...     y *= 1e3
...     x, y = pyproj.transform(proj_gk4, proj_wgs84, x, y)
...     return x, y, z  # doctest: +SKIP

Then, we can load the NonLinLoc Hypocenter-Phase file into an ObsPy
:class:`~obspy.core.event.Catalog` object using
:func:`~obspy.core.event.readEvents`, supplying our coordinate mapping function
as `coordinate_converter` kwarg, which will be passed down to the low-level
routine :func:`~obspy.nlloc.core.read_nlloc_hyp`.

>>> from obspy import readEvents
>>> cat = readEvents("/path/to/nlloc.hyp",
...                  coordinate_converter=my_conversion)  # doctest: +SKIP
>>> print(cat)  # doctest: +SKIP
1 Event(s) in Catalog:
2010-05-27T16:56:24.612600Z | +48.047,  +11.646

>>> event = cat[0]  # doctest: +SKIP
>>> print(event)  # doctest: +SKIP
Event:  2010-05-27T16:56:24.612600Z | +48.047,  +11.646
<BLANKLINE>
       resource_id: ResourceIdentifier(id="smi:local/...")
     creation_info: CreationInfo(creation_time=..., version='NLLoc:v6.00.0')
    ---------
             picks: 8 Elements
           origins: 1 Elements

>>> origin = event.origins[0]  # doctest: +SKIP
>>> print(origin)  # doctest: +SKIP
    Origin
                resource_id: ResourceIdentifier(id="smi:local/...")
                       time: UTCDateTime(2010, 5, 27, 16, 56, 24, 612600)
                  longitude: 11.64553754...
                   latitude: 48.04707051...
                      depth: 4579.4... [confidence_level=68,
 uncertainty=191.6063...]
                        ...
              creation_info: CreationInfo(creation_time=UTCDateTime(2014, 10,
17, 16, 30, 8), version='NLLoc:v6.00.0')
        ---------
                   comments: 1 Elements
                   arrivals: 8 Elements
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
