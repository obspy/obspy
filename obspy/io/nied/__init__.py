"""
obspy.io.nied - NIED's moment tensors TEXT format support for ObsPy
================================================================

This module provides read support for the moment tensor files (TEXT format)
provided by the National Research Institute for Earth Science and Disaster
Prevention in Japan (NIED; http://www.fnet.bosai.go.jp/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Example
-------

It works by utilizing ObsPy's :func:`~obspy.core.event.read_events` function.

>>> import obspy
>>> cat = obspy.read_events("/path/to/NIEDCATALOG")
>>> print(cat)
1 Event(s) in Catalog:
2011-03-11T05:46:18.120000Z | +38.103, +142.861 | 9.0 ML

The event will contain a couple of origins and magnitudes.

>>> print(cat[0])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
Event:  2011-03-11T05:46:18.120000Z | +38.103, +142.861 | 9.0 ML
<BLANKLINE>
              resource_id: ResourceIdentifier(id="...")
               event_type: 'earthquake'
        ---------
         focal_mechanisms: 1 Elements
                  origins: 2 Elements
               magnitudes: 2 Elements

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
