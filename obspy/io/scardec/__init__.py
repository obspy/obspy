#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.io.scardec - SCARDEC file format support for ObsPy
================================================================

This module provides read/write support for the SCARDEC source
file format supplied by IPGP.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Example
-------

It works by utilizing ObsPy's :func:`~obspy.core.event.read_events` function.

>>> import obspy
>>> cat = obspy.read_events("/path/to/SCARDEC_file")
>>> print(cat)
1 Event(s) in Catalog:
2003-12-26T01:56:58.129999Z | +29.100,  +58.240 | 6.54 mw

The event will contain one origins with a moment rate function.

>>> print(cat[0])  # doctest: +NORMALIZE_WHITESPACES +ELLIPSIS
Event:  2014-01-25T05:14:18.000000Z |  -7.985, +109.265 | 6.202 mw

               resource_id: ResourceIdentifier(id='...')
                event_type: 'earthquake'
    ---------
        event_descriptions: 1 Elements
                  comments: 1 Elements
          focal_mechanisms: 1 Elements
                   origins: 1 Elements
                magnitudes: 1 Elements
     source_time_functions: 1 Elements


This module also offers write support for the SCARDEC format.

>>> cat.write("output/SCARDEC_file", format='SCARDEC')  # doctest: +SKIP

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
