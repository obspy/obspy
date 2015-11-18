#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.io.stationtxt - Read support for the FDSNWS station text files
====================================================================

This module provides read support for the station text files served by the FDSN
web services.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


Example
-------

Don't use this module directly but utilize it through the
:func:`~obspy.core.inventory.inventory.read_inventory` function.

>>> import obspy
>>> inv = obspy.read_inventory("/path/to/channel_level_fdsn.txt")
>>> print(inv)  # # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
Inventory created at ...
    Created by: ObsPy ...
            http://www.obspy.org
    Sending institution: None
    Contains:
        Networks (2):
            AK
            AZ
        Stations (3):
            AK.BAGL ()
            AK.BWN ()
            AZ.BZN ()
        Channels (6):
            AK.BAGL..LHZ, AK.BWN..LHZ, AK.BWN..LHZ, AZ.BZN..LHZ, AZ.BZN..LHZ,
            AZ.BZN..LHZ
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
