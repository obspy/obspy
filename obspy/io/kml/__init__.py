# -*- coding: utf-8 -*-
"""
obspy.io.kml - Keyhole Markup Language (KML) write support
==========================================================
This module provides write support for the Keyhole Markup Language (KML)
format.

Write support works via the ObsPy plugin structure for
:class:`~obspy.core.event.Catalog` and
:class:`~obspy.core.inventory.inventory.Inventory`:

>>> from obspy import read_inventory, read_events
>>> inv = read_inventory()  # load example data
>>> inv.write("my_stations.kml", format="KML")  # doctest: +SKIP
>>> cat = read_events()  # load example data
>>> cat.write("my_events.kml", format="KML")  # doctest: +SKIP

For details on further parameters see
:meth:`~obspy.io.kml.core.inventory_to_kml_string` and
:meth:`~obspy.io.kml.core.catalog_to_kml_string`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
