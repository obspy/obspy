# -*- coding: utf-8 -*-
"""
obspy.io.shapefile - ESRI shapefile write support for ObsPy
===========================================================
This module provides write support for the ESRI shapefile format.

Write support works via the ObsPy plugin structure for
:class:`~obspy.core.event.Catalog` and
:class:`~obspy.core.inventory.inventory.Inventory`:

>>> from obspy import read_inventory, read_events
>>> inv = read_inventory()  # load example data
>>> inv.write("my_stations.shp", format="SHAPEFILE")  # doctest: +SKIP
>>> cat = read_events()  # load example data
>>> cat.write("my_events.shp", format="SHAPEFILE")  # doctest: +SKIP

Additional information can be written to the shapefile as custom
database columns. In this toy example we add the Flinn Engdahl region as a
database column (see :func:`obspy.io.shapefile.core._write_shapefile()`):

>>> from obspy.geodetics.flinnengdahl import FlinnEngdahl
>>> fe = FlinnEngdahl()
>>> regions = [
...     fe.get_region(event.origins[0].longitude, event.origins[0].latitude)
...     for event in cat]
>>> extra_fields = [('Region', 'C', 100, None, regions)]
>>> cat.write("my_events.shp", format="SHAPEFILE",
...           extra_fields=extra_fields)  # doctest: +SKIP

Note that the number of values given for each custom database column must be
equal to the number of events in a given catalog or equal to the total number
of stations combined across all networks in a given inventory.

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
