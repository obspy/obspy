# -*- coding: utf-8 -*-
"""
obspy.io.shapefile - ESRI shapefile write support for ObsPy
===========================================================
This module provides write support for the ESRI shapefile format.

Write support works via the ObsPy plugin structure for
:class:`~obspy.core.event.Catalog` and
:class:`~obspy.core.inventory.Inventory`:

>>> from obspy import read_inventory, read_events
>>> inv = read_inventory()  # load example data
>>> inv.write("my_stations.shp", format="SHAPEFILE")  # doctest: +SKIP
>>> cat = read_events()  # load example data
>>> cat.write("my_events.shp", format="SHAPEFILE")  # doctest: +SKIP

For Catalog objects, an additional shapefile with error ellipses representing
the origin uncertainty can be written using the ``error_ellipses`` kwarg. Since
origin uncertainties are in meters (as opposed to degrees), an appropriate
coordinate system has to be specified by EPSG code (the example contains events
near Vanuatu and correspondingly the EPSG code for 'WGS 84 / UTM zone 58S'):

>>> cat = read_events('/path/to/vanua.sum.grid0.loc.hyp')  # doctest: +SKIP
>>> cat.write('my_events.shp', format='SHAPEFILE',  # doctest: +SKIP
...           error_ellipses={'filename': 'my_events_errors.shp',
...                           'epsg': 32758})


.. seealso::

    The format definition can be found
    `here <https://www.esri.com/library/whitepapers/pdfs/shapefile.pdf>`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
