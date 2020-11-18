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
>>> cat = obspy.read_events("/path/to/test.scardec")
>>> print(cat)
1 Event(s) in Catalog:
2014-01-25T05:14:18.000000Z |  -7.985, +109.265 | 6.20 mw

The event will contain one origins with a moment rate function.

>>> print(cat[0])  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
Event:  2014-01-25T05:14:18.000000Z |  -7.985, +109.265 | 6.20 mw
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
                         origins: 1 Elements
                      magnitudes: 1 Elements


This module also offers write support for the SCARDEC format.

>>> cat.write("output/SCARDEC_file", format='SCARDEC')  # doctest: +SKIP

"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
