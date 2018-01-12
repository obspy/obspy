#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.io.cmtsolution - CMTSOLUTION file format support for ObsPy
================================================================

This module provides read/write support for the CMTSOLUTION files used by
many waveform solvers.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Example
-------

It works by utilizing ObsPy's :func:`~obspy.core.event.read_events` function.

>>> import obspy
>>> cat = obspy.read_events("/path/to/CMTSOLUTION")
>>> print(cat)
1 Event(s) in Catalog:
2003-12-26T01:56:58.130000Z | +29.100,  +58.240 | 6.54 mw

The event will contain a couple of origins and magnitudes.

>>> print(cat[0])  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
Event:     2003-12-26T01:56:58.130000Z | +29.100,  +58.240 | 6.54 mw
<BLANKLINE>
                     resource_id: ResourceIdentifier(id="...")
                      event_type: 'earthquake'
             preferred_origin_id: ResourceIdentifier(id="...")
          preferred_magnitude_id: ResourceIdentifier(id="...")
    preferred_focal_mechanism_id: ResourceIdentifier(id="...")
                            ---------
              event_descriptions: 1 Elements
                        comments: 1 Elements
                focal_mechanisms: 1 Elements
                         origins: 2 Elements
                      magnitudes: 3 Elements

This module also offers write support for the CMTSOLUTION format.

>>> cat.write("output/CMTSOLUTION")  # doctest: +SKIP

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
